"""Input/output helpers for the CompMusic Indian Art Music Raga Recognition
Dataset (mirdata id "compmusic_raga"), Hindustani subset (HMD).

Why this does not use mirdata's track loader
--------------------------------------------
mirdata ships a `compmusic_raga` loader, but its v1.0 index covers ONLY the
477 Carnatic recordings -- `dataset.track_ids` yields zero Hindustani tracks
even though `dataset.download()` fetches an archive containing both
traditions. So mirdata is used for the download only, and the Hindustani
half is read straight off the extracted tree.

Layout of the extracted Hindustani subset
-----------------------------------------
    RagaDataset/Hindustani/
        _info_/path_mbid_ragaid.txt              path <TAB> mbid <TAB> raga_id
        _info_/ragaId_to_ragaName_mapping.json   raga_id -> display name
        features/<raga_id>/<artist>/<album>/<title>_<mbid>.pitch
                                                  ... .tonic

Tracks are joined to metadata by MBID (the UUID suffix present on every
feature filename) rather than by the path recorded in the metadata: 64 of
the 300 recorded paths do not match what is on disk (album-name
mismatches), whereas the MBID join resolves all 300.

Audio is never touched -- it is ~9.2 TB and access-restricted, and the
feature pipeline only consumes pitch/tonic arrays.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from ..core import PitchContour

DEFAULT_DATA_HOME = os.path.expanduser("~/mir_projects/data/compmusic_raga")

# Every feature filename ends in the recording's MusicBrainz id.
MBID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$"
)


class CompMusicHindustani:
    """Adapter over the CompMusic HMD subset (300 tracks, 30 ragas).

    Mirrors `io.saraga.SaragaHindustani`, so anything written against one
    works against the other.
    """

    name = "compmusic_hmd"

    def __init__(self, data_home: str | None = None, download: bool = False):
        self.data_home = data_home or os.environ.get(
            "COMPMUSIC_RAGA_DATA_HOME", DEFAULT_DATA_HOME
        )
        self.root = Path(self.data_home) / "RagaDataset" / "Hindustani"

        if download and not self.root.exists():
            self.download()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Hindustani subset not found at {self.root}. "
                f"Construct with download=True to fetch it (~3.4 GB)."
            )

        self._records: list[dict] | None = None

    def download(self) -> None:
        """Fetch the features archive via mirdata.

        Only the free features/metadata zip is fetched -- the dataset's
        `remotes` contains a single "features" entry, so this cannot reach
        for the access-restricted audio.
        """
        import mirdata

        mirdata.initialize("compmusic_raga", data_home=self.data_home).download()

    def _raga_names(self) -> dict[str, str]:
        """Map raga_id -> human-readable raga name."""
        path = self.root / "_info_" / "ragaId_to_ragaName_mapping.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _build_records(self) -> list[dict]:
        raga_names = self._raga_names()

        # Index what is actually on disk by mbid, since the metadata's paths
        # are not always accurate (see module docstring).
        pitch_by_mbid: dict[str, Path] = {}
        for pitch_path in self.root.glob("features/*/*/*/*.pitch"):
            match = MBID_RE.search(pitch_path.stem)
            if match:
                pitch_by_mbid[match.group(1)] = pitch_path

        meta_path = self.root / "_info_" / "path_mbid_ragaid.txt"
        records = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                _path, mbid, raga_id = line.split("\t")
                pitch_path = pitch_by_mbid.get(mbid)
                if pitch_path is None:
                    continue
                records.append(
                    {
                        "track_id": mbid,
                        "raga_id": raga_id,
                        "raga_label": raga_names.get(raga_id, raga_id),
                        "artist": pitch_path.parts[-3],
                        "pitch_path": pitch_path,
                        "tonic_path": pitch_path.with_suffix(".tonic"),
                    }
                )
        return records

    def list_tracks(self) -> list[dict]:
        """Return one record per recording, with track_id and raga_label.

        Records also carry raga_id, artist, and the resolved feature paths,
        which the pipeline ignores but inspection scripts find useful.
        """
        if self._records is None:
            self._records = self._build_records()
        return self._records

    def _record(self, track_id: str) -> dict:
        for record in self.list_tracks():
            if record["track_id"] == track_id:
                return record
        raise KeyError(f"Unknown track_id {track_id!r}")

    def get_pitch(self, track_id: str) -> PitchContour:
        """Read a .pitch file into a PitchContour.

        The .pitch files are two-column TSVs (time_seconds, frequency_hz)
        with 0.0 Hz marking unvoiced frames -- the same convention
        PitchContour applies when an F0Data carries no explicit voicing.
        """
        record = self._record(track_id)
        data = np.loadtxt(record["pitch_path"], delimiter="\t")
        freqs = data[:, 1]

        return PitchContour(
            times=data[:, 0],
            frequencies=freqs,
            voiced_mask=freqs > 0,
            source=self.name,
            track_id=track_id,
        )

    def get_tonic(self, track_id: str) -> float | None:
        """Return the dataset's annotated tonic in Hz, or None if absent."""
        path = self._record(track_id)["tonic_path"]
        if not path.exists():
            return None
        return float(path.read_text().strip())
