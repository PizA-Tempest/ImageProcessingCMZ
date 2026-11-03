import sqlite3, os
p = "outputs/single/CAM3_Dataset3.sqlite"
print("size", os.path.getsize(p))
con = sqlite3.connect(p)
print("tables:", con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print("count:", con.execute("SELECT COUNT(*) FROM behavior_event").fetchone())
rows = con.execute("SELECT behavior, timestamp_start, timestamp_end FROM behavior_event LIMIT 10").fetchall()
for r in rows:
    print(" ", r)
