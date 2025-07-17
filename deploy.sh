#!/bin/bash

# Get API keys from command line arguments
COHERE_API_KEY=$1
TAVILY_API_KEY=$2

# Navigate to the application directory (replace with your actual path if different)
cd /home/ec2-user/Podcast-Q-A-Assistant # Amazon Linux default user is ec2-user
# cd /home/ubuntu/your-repo-name # For Ubuntu instances

# Pull the latest code from GitHub
echo "Pulling latest code from GitHub..."
git pull origin master # Adjust 'main' if your default branch is different



# Create/update config.py with the provided API keys
echo "Generating config.py..."
cat > config.py <<EOL
# Auto-generated during deployment
COHERE_API_KEY = "$COHERE_API_KEY"
TAVILY_API_KEY = "$TAVILY_API_KEY"
EOL


# Install/update Python dependencies (optional, but good for new packages)
echo "Installing/updating Python dependencies..."
pip3 install -r requirements.txt

# Display done installing
echo "DONE INSTALLING"

# Restart the Streamlit app
echo "Restarting Streamlit app..."
# Find and kill any existing streamlit processes on port 8501
sudo lsof -t -i:8501 | xargs -r kill -9
# Start the Streamlit app in the background using nohup
nohup streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false > streamlit_app.log 2>&1 &

echo "Deployment complete."