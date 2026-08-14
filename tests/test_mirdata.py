import os
import mirdata

DATA_HOME = os.path.expanduser("~/mir_projects/data")

saraga = mirdata.initialize(
    "saraga_hindustani",
    data_home=DATA_HOME
)

saraga.download(partial_download=["index"])

print("Track count:", len(saraga.track_ids))
print("First 5 IDs:", saraga.track_ids[:5])

track = saraga.choice_track()
print("Track ID:", track.track_id)
print("Title:", track.metadata["title"])
print("Raags:", [r["common_name"] for r in track.metadata["raags"]])
print("Artists:", [a["artist"]["name"] for a in track.metadata["artists"]])
