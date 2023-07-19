#!/bin/bash

# navigate to the 'final_submission_tn_processed' directory
cd final_submission_tn_processed

# find and rename files
find . -type f -name "*.json" -print0 | while IFS= read -r -d '' file; do
    # extract the sessionID from the file name
    sessionID=$(echo "$file" | sed -r 's|.*/[^-]*-[^-]*-([^/]*)\.json|\1|g')
    # rename the file to <sessionID>.json
    mv -- "$file" "./$(dirname "$file")/$sessionID.json"
done
