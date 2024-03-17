## Video Question Answering Bot Readme

### Overview
This AI bot is designed to accept a video URL as input and answer questions related to the content of the video. Utilizing various frameworks such as Langchain, the bot extracts transcripts from the video, tokenizes them, preprocesses the data, and feeds it into a pre-trained Large Language Model (LLM) accessed via API.

### Features
- **Video Question Answering**: The bot can answer questions based on the content of the provided video.
- **Framework Integration**: Utilizes frameworks like Langchain to process and analyze video transcripts.
- **API Access**: Accesses pre-trained LLMs through OpenAI's API for enhanced question answering capabilities.

### Dependencies
- **Python**: Ensure you have the latest version of Python installed, which can be obtained from [python.org/downloads](https://www.python.org/downloads/).
- **OpenAI API Key**: Obtain an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) and paste it into the specified location in `secret_key.py` to access the pre-trained LLM.

### Note
- Only new users without an existing OpenAI account are eligible for the free trial. You will need a fresh email ID and mobile number without any previous OpenAI association to access the API key for free trial access.

### Installation and Setup
1. **Python IDE**: Choose a Python Integrated Development Environment (IDE) such as Visual Studio Code, IDLE, or PyCharm for running the code.
2. **Install Dependencies**: Install the required modules listed in `requirements.txt` using the following command:
   ```
   pip install -r requirements.txt
   ```
3. **API Key Setup**: Copy the OpenAI API key from the website and paste it into `secret_key.py`.
4. **Local Deployment**: Navigate to the cloned repository directory and open the terminal. Run the following command to start the application on localhost:
   ```
   streamlit run VideExp.py
   ```
5. **Usage**: Once the application is running locally, input the video URL and ask questions related to the content. The bot will provide answers based on the video's transcript.

### Purpose
This application is developed for educational purposes and is designed to integrate seamlessly with e-learning websites.

### Happy Learning!
Now you're ready to dive in! Ask away about the videos and explore the depths of knowledge. Go crack some eggs! 🥚📚
