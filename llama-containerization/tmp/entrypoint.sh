#!/bin/sh

# Check if DOWNLOAD_LATER is set to false
if [ "$DOWNLOAD_LATER" = "false" ]; then
  # Run the Python script and capture its output as the environment variable
  # If there's a script to download the model, execute it here, and then export the path
  # Example: python download_model.py
  export MODEL_LOCAL_PATH="$APP_DIR/checkpoints/consolidated.00.pth"

  # Check if MODEL_LOCAL_PATH is empty
  if [ -z "$MODEL_LOCAL_PATH" ]; then
    echo "Model download failed or did not produce output. Exiting..."
    exit 1
  fi
else
  echo "Skipping model download as DOWNLOAD_LATER is set to true."
fi

export MODEL_LOCAL_PATH="$APP_DIR/checkpoints/consolidated.00.pth"

# Execute the Python script
#echo "Running the Python application..."
#exec python app.py
