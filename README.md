## Interactive Laptop Recommendation Assistant Chatbot

---

### Project Introduction
The **Interactive Laptop Recommendation Assistant Chatbot** is an AI-driven assistant that helps users select the best laptop based on their budget, purpose, and preferences. It features natural language interaction, image-based laptop identification, and personalized recommendations to streamline the laptop-buying process.

#### Key Features:
- **AI-powered Laptop Recommendations** - Get suggestions based on budget, usage (gaming, work, study), portability, and brand.
- **Laptop Image Recognition** - Upload an image of a laptop for model identification and analysis.
- **Data-Driven Insights** - View comparison tables, pricing trends, and historical recommendations.
- **Advanced Filtering** - Refine search results by specific preferences like RAM, GPU, weight, and battery life.
- **Secure User Authentication** - Create an account to save search history and track past recommendations.

---

### Project Demo
For detailed instructions, see the **User's Guide** section in the `Final Report.pdf`, or watch the `demo.mp4`

---

### Installation & Setup

Follow these steps to clone and set up the project:

#### Step 1: Clone the Repository
```
git clone https://github.com/Kairui-Zhang/Interactive-Laptop-Recommendation-Assistant-Chatbot.git
cd Interactive-Laptop-Recommendation-Chatbot-main
```

#### Step 2: Create and Activate a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

#### Step 3: Fill in Your API Key
To ensure the proper functionality of the application, you need to fill in your API key in `app.py`.
```
# API Configuration
API_KEY = "" # FILL IN YOUR OWN API KEY HERE
client = OpenAI(api_key=API_KEY)
```


#### Step 4: Install Dependencies
```
pip install -r requirements.txt
```

#### Step 5: Run the Flask Server
```
python app.py
```
Access the chatbot at: `http://127.0.0.1:5000/`

---

### System Architecture

This chatbot is a **Flask-based AI system** that integrates:
- **Backend**: Flask, SQLite
- **Frontend**: HTML, CSS, JavaScript
- **AI Integration**: OpenAI GPT-4 API
- **Visualization**: Chart.js for price, specification comparison

---

### Project Structure
```
Interactive Laptop Recommendation Chatbot/
├── Image recognition datasets for laptops/
├── templates/
│   ├── index.html
│   ├── login.html
│   └── result.html
├── app.py
├── categorize.py
├── chat_history.db
├── demo.mp4
├── Final Report.pdf
├── laptop_advisor.db
├── laptop_recommendations_2.db
├── laptop_recommendations_3.db
├── laptop_recommendations_4.db
├── Presentation Slides.pdf
├── README.md
├── requirements.txt
└── test.py
```

---

### Team Members
This project is part of the ***Foundations and Application of Generative AI*** at **Technical University of Munich**. The team is composed of the following members:

- **Lingwei Lu (MMDT)**
  - Team Lead
  - Full-Stack Developer
  - Project Management & Requirements
- **Kairui Zhang (MMDT)**
  - AI Specialist
  - Backend Developer
  - Presentation Strategist
- **Haojun Liang (BIE)**
  - Data Retriever
  - Backend Developer
- **XiaoYao Wang (BIE)**
  - Debugging & Testing
  - Frontend Development
