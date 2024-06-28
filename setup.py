import os
import sys

# Command to run
command = ["npm install -g npx","npm install --save-dev electron","npm init"]

for c in command:
    os.system(command)