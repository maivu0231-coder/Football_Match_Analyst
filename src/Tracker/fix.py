with open(r'd:\Python AI\Personal_Project\Football_Analyst_Project\Tracker\Tracker.py', 'rb') as f:
    data = f.read()

# Replace lines 44-48 safely
import re
pattern = rb'ball_positions = \[.*?\]\r?\n.*?return tracks'
replacement = b'ball_positions = [{1: {"bbox":x}} for x in df_ball_postions.to_numpy().tolist()]\r\n        tracks["ball"] = ball_positions\r\n        return tracks'

new_data, count = re.subn(pattern, replacement, data, flags=re.DOTALL)
print("Sub count:", count)
with open(r'd:\Python AI\Personal_Project\Football_Analyst_Project\Tracker\Tracker.py', 'wb') as f:
    f.write(new_data)
