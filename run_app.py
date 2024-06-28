import os
import sys

# Command to run
command = ["npm run start"]

for c in command:
    os.system(command)

if sys.platform == "win32":
    os.system("pause")