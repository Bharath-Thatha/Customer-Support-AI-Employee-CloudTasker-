# CloudTasker Support Bot

A support chatbot application built with Streamlit and LangChain for the CloudTasker platform. It intelligently categorizes user queries and uses a vector database to fetch relevant information to assist users with billing, account access, or technical issues.

## Setup Instructions

1. **Clone or Download the Repository**
   Ensure you have the project files (like `app.py` and `faq_data.json`) in your local directory.

2. **Set up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install streamlit python-dotenv langchain-openai langchain-huggingface langchain-community langchain-core chromadb numpy
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your required API keys (e.g., OpenAI API Key).
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the Application**
   Start the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

## Assumptions Made

During the development and design of this application, the following assumptions were made:

1. **Greetings Handling:** Greetings (like "hello", "hi") are not treated as a specific category. The bot will respond to greetings appropriately, but they are not classified into a functional category (like billing, account access, etc.) for vector search.

2. **Categorized Vector Search:** When user input is received, it is first classified into one of the core categories (e.g., `Billing`, `Account Access`, or `Technical`). Subsequently, the application only searches the vector knowledge base that pertains specifically to that identified category. This ensures focused and relevant responses.

3. **Escalation to Human Agent:** If the information required to answer the user's query is not present within the `faq_data.json` knowledge base, the bot is designed to gracefully escalate the issue to a human support agent instead of hallucinating an answer.
