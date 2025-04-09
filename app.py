# Standard libraries
import os
import re
import json
import sqlite3
import traceback
import base64
import io

# Third-party libraries
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from PIL import Image
import requests

# ------------------------------------------------------------

# Absolute path for the database
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path
chat_history_db_path = os.path.join(BASE_DIR, 'chat_history.db')
laptop_advisor_db_path = os.path.join(BASE_DIR, 'laptop_recommendations_3.db')

app = Flask(__name__)
CORS(app)
app.secret_key = '88888888'

# Configure session cookies to enhance security
app.config['SESSION_COOKIE_SECURE'] = True # Ensure cookies are transmitted only over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True # Prevent client-side scripts from accessing cookies

# API Configuration
API_KEY = "" # FILL IN YOUR OWN API KEY HERE
client = OpenAI(api_key=API_KEY)

# ------------------------------------------------------------

# Process and analyze user input
def parse_user_input(user_message, conversation_history):
    """
    Decision flow:
    1) If the intent is "recommendation":
       - First, attempt generate_recommendations_based_on_state.
       - If the result contains "couldn't find any laptops" or the state is incomplete, 
         then try fallback_recommendation_system.
       - If there are still no results, call GPT.
    2) If the intent is not "recommendation", directly call GPT.
    """  
    
    try:
        # Handle numeric selection
        if user_message.strip() in ['1', '2', '3']:
            if 'recommendation_links' in session:
                selected = user_message.strip()
                link = session['recommendation_links'].get(selected)
                if link: # Only check if it exists
                    return f"🔗 Purchase Link for Option {selected}:\n{link}"
                else:
                    return "⚠️ Sorry, the purchase link is currently unavailable for this option."
            else:
                return "ℹ️ Please start a recommendation session first."
        
        # Check if it's an image analysis request
        if '[IMAGE_ANALYSIS_REQUEST]' in user_message:
            # If there is a current image analysis result, return it directly
            if 'current_analysis' in session and session['current_analysis']:
                return session['current_analysis']
            return "I apologize, but I couldn't analyze the image. Please try uploading it again."
            
        # Determine user intent
        user_intent = analyze_user_intent(user_message, conversation_history)
        
        # If the message contains keywords "analyze" and "image"
        if "analyze" in user_message.lower() and "image" in user_message.lower():
            if 'current_analysis' in session and session['current_analysis']:
                return session['current_analysis']
            return "Please upload an image for me to analyze."

        # If the user's intent is "recommendation"
        if user_intent.get('type') == 'recommendation':
            # ------------------------------
            # 1A) Parse budget (string -> int)
            # ------------------------------
            purpose_value = user_intent.get('purpose')
            budget_str = user_intent.get('budget')
            
            if not purpose_value and not budget_str:
                return (
                    "Sure! I'd love to help you pick a laptop.\n"
                    "Could you share more details about what you'll use it for (e.g., gaming, work, study)?\n"
                    "and what your approximate budget is?\n"
                    "This will help me recommend something that truly fits your needs!\n"
                )
            
            budget_value = 2000
            if budget_str:
                match = re.match(r'\$?(\d+)(?:-(\d+))?', budget_str)
                if match:
                    if match.group(2):
                        budget_value = int(match.group(2))
                    else:
                        budget_value = int(match.group(1))

            # ------------------------------
            # 1B) Parse purpose
            # ------------------------------
            if not purpose_value:
                purpose_value = "General"

            # ------------------------------
            # 1C) Call generate_recommendations_based_on_state
            # ------------------------------
            state = {
                'purpose': purpose_value,
                'budget': budget_value,
                'portability': None,
                'brand': None
            }
            db_result = generate_recommendations_based_on_state(state)

            # Check if generate_recommendations_based_on_state did not find results
            # If it returns "I couldn't find any laptops" or similar prompts, assume no results were found
            if "couldn't find any laptops" in db_result.lower():
                # Try fallback_recommendation_system
                fallback_result = fallback_recommendation_system(user_message, conversation_history)

                # If fallback also fails
                if "I couldn't find" in fallback_result.lower() or "trouble accessing" in fallback_result.lower():
                    # As a last resort, let GPT answer
                    return call_gpt(user_message, conversation_history)
                else:
                    # If fallback returns results, return them
                    return fallback_result
            else:
                # If generate_recommendations_based_on_state succeeds, return the result
                return db_result

        # ------------------------------
        # If not a recommendation request, call GPT
        # ------------------------------
        return call_gpt(user_message, conversation_history)

    except Exception as e:
        print(f"Error in parse_user_input: {e}")
        report_error()
        return "I apologize, but I encountered an error. Could you please rephrase your question?"

# ------------------------------------------------------------

# Analyze user intent
def analyze_user_intent(message, history):
    intent = {
        'type': None,
        'purpose': None,
        'budget': None,
        'priority': None
    }
    
    message_lower = message.lower()
    
    # Check if it is a recommendation request
    if any(word in message_lower for word in ['recommend', 'suggest', 'buy', 'looking for', 'want']):
        intent['type'] = 'recommendation'
        
        # Analyze usage purpose
        if 'gaming' in message_lower or 'game' in message_lower:
            intent['purpose'] = 'Gaming'
        elif 'work' in message_lower or 'business' in message_lower:
            intent['purpose'] = 'Work'
        elif 'student' in message_lower or 'study' in message_lower:
            intent['purpose'] = 'Study'
            
        # Analyze budget
        budget_match = re.search(r'\$?(\d+)(?:-(\d+))?', message)
        if budget_match:
            if budget_match.group(2): # Range budget
                intent['budget'] = f"${budget_match.group(1)}-${budget_match.group(2)}"
            else:
                intent['budget'] = f"${budget_match.group(1)}"
                
    # Check if it is an analysis request
    elif 'analyze' in message_lower or 'what about' in message_lower:
        intent['type'] = 'analysis'
        
    return intent

# ------------------------------------------------------------

# Call GPT
def call_gpt(user_message, conversation_history):
    # Encapsulate the GPT call separately for easy invocation in parse_user_input()

    # More intelligent system prompt
    system_prompt = """You are an intelligent and helpful laptop advisor. Important guidelines:
        1. Be conversational and natural in your responses
        2. Understand user intent beyond just the words - consider the context of the entire conversation
        3. If a user changes topics (e.g., from analyzing a specific laptop to seeking recommendations), adapt smoothly
        4. Provide personalized advice based on the user's needs and preferences
        5. When making recommendations:
           - Consider the user's stated budget and requirements
           - Explain why each recommendation would be suitable
           - Be specific about features and benefits
        6. Keep responses concise but informative
        7. Use a friendly, engaging tone while maintaining professionalism
        8. Strictly focus on providing laptop recommendations and related advice
        9. Do not respond to requests that are unrelated to laptops or computer technology
        10. If the user asks for unrelated content, politely decline and redirect the conversation to laptop recommendations"""
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation history based on existing logic
    user_intent = analyze_user_intent(user_message, conversation_history)
    if conversation_history:
        relevant_history = get_relevant_history(conversation_history, user_intent)
        messages.extend(relevant_history)

    # Finally, add the current user query
    messages.append({"role": "user", "content": user_message})

    # Get GPT response
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.8,
        max_tokens=1000
    )
    return completion.choices[0].message.content

# ------------------------------------------------------------

# Retrieve relevant conversation history
def get_relevant_history(history, current_intent):
    relevant_messages = []
    
    if current_intent['type'] == 'recommendation':
        # Keep only history related to recommendations
        for msg in history[-5:]:
            if any(word in msg['content'].lower() for word in ['recommend', 'suggest', 'buy', 'budget', 'gaming', 'work']):
                relevant_messages.append(msg)
    else:
        # For other types of requests, retain only the most recent messages
        relevant_messages = history[-3:]
        
    return relevant_messages

# ------------------------------------------------------------

# Check if the input is a specific question rather than part of a general conversation flow
def is_specific_question(user_input): 
    specific_keywords = [
        'cpu', 'processor', 'gpu', 'graphics', 'ram', 'memory',
        'storage', 'screen', 'display', 'battery', 'weight',
        'what is', 'how much', 'tell me about', 'can you explain'
    ]
    
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in specific_keywords)

# ------------------------------------------------------------

def analyze_purpose(user_input):
    input_lower = user_input.lower()
    if any(word in input_lower for word in ["game", "gaming", "play"]):
        return "Gaming"
    elif any(word in input_lower for word in ["work", "business", "office"]):
        return "Business"
    elif any(word in input_lower for word in ["study", "student", "school"]):
        return "Student"
    elif any(word in input_lower for word in ["general", "daily", "regular"]):
        return "General"
    return None

# ------------------------------------------------------------

def analyze_budget(user_input):
    input_lower = user_input.lower()
    # Attempt to extract numbers
    numbers = re.findall(r'\$?(\d+)', input_lower)
    if numbers:
        budget = int(numbers[0])
        if 300 <= budget <= 5000: # Reasonable budget range
            return budget
    return None

# ------------------------------------------------------------

def analyze_portability(user_input):
    input_lower = user_input.lower()
    if any(word in input_lower for word in ["light", "portable", "thin", "carry"]):
        return "Portable"
    elif any(word in input_lower for word in ["heavy", "desktop", "powerful", "gaming"]):
        return "Desktop Replacement"
    return None

# ------------------------------------------------------------

def analyze_brand(user_input):
    input_lower = user_input.lower()
    brands = {
        "dell": "Dell",
        "hp": "HP",
        "lenovo": "Lenovo",
        "asus": "Asus",
        "acer": "Acer",
        "apple": "Apple"
    }
    for brand_lower, brand_proper in brands.items():
        if brand_lower in input_lower:
            return brand_proper
    return None

# ------------------------------------------------------------

# Main recommendation system
def generate_recommendations_based_on_state(state):
    try:
        conn = sqlite3.connect(laptop_advisor_db_path)
        cursor = conn.cursor()
        
        query = '''
        SELECT model, price, weight, gpu, ram, tags, link
        FROM laptops
        WHERE price <= ?
        AND (tags LIKE ? OR tags LIKE ?)
        '''
        params = [state['budget'], f"%{state['purpose']}%", "%Gaming%"]
        
        if state['portability'] == "Portable":
            query += " AND weight <= 2.5"
        
        if state['brand']:
            query += " AND model LIKE ?"
            params.append(f"%{state['brand']}%")
        
        query += " ORDER BY RANDOM() LIMIT 3"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "I couldn't find any laptops matching your specific requirements. Would you like to adjust some criteria?"
            
        # Format each recommendation more clearly
        response = [
            "\n🔍 Based on your preferences:",
            f"• Purpose: {state['purpose']}",
            f"• Budget: ${state['budget']}",
            f"• Portability: {state['portability']}",
            f"• Brand: {state['brand'] or 'Any'}",
            "\n📌 Here are my recommendations:\n" # Add a blank line after each recommendation
        ]
        
        session['recommendation_links'] = {}
        
        # Format each recommendation more clearly
        for i, (model, price, weight, gpu, ram, tags, link) in enumerate(results, 1):
            session['recommendation_links'][str(i)] = link # Store purchase link
            response.extend([
                f"💻 Recommendation #{i}:",
                f"  • Model: {model}",
                f"  • Price: ${price:,.2f}",
                f"  • Weight: {weight}kg",
                f"  • GPU: {gpu}",
                f"  • RAM: {ram}GB",
                f"  • Features: {tags}",
                "" # Add a blank line after each recommendation
            ])
        
        response.append("\n💭 Would you like to know more about any of these laptops? Or type 'different options' for more recommendations.")
        response.append("\n💭 Or else type 1, 2, or 3 to select a laptop and get the purchase link.")
        
        # Use double newlines to create a clearer separation
        return "\n".join(response)
        
    except Exception as e:
        print(f"Recommendation error: {e}")
        return "I apologize, but I'm having trouble generating recommendations right now."

# ------------------------------------------------------------

# Backup recommendation system
def fallback_recommendation_system(user_input, conversation_history):
    """备用推荐系统，基于关键词匹配和数据库查询"""
    try:
        # Analyze key information from user input
        input_lower = user_input.lower()
        
        # Extract usage purpose
        purpose = "General"
        if any(word in input_lower for word in ["game", "gaming", "play"]):
            purpose = "Gaming"
        elif any(word in input_lower for word in ["work", "business", "office"]):
            purpose = "Business"
        elif any(word in input_lower for word in ["student", "study", "school"]):
            purpose = "Student"
            
        # Extract budget
        budget = 2000 # Default budget
        budget_keywords = {
            "cheap": 800,
            "budget": 800,
            "expensive": 3000,
            "high-end": 3000,
            "mid": 1500,
            "middle": 1500
        }
        for keyword, value in budget_keywords.items():
            if keyword in input_lower:
                budget = value
                break
                
        # Retrieve recommendations from the database
        conn = sqlite3.connect(laptop_advisor_db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT model, price, weight, gpu, ram, tags, link
            FROM laptops
            WHERE price <= ? 
            AND (
                tags LIKE ? 
                OR tags LIKE ? 
                OR tags LIKE ?
            )
            ORDER BY 
                CASE 
                    WHEN tags LIKE ? THEN 1
                    WHEN tags LIKE ? THEN 2
                    ELSE 3
                END,
                price DESC
            LIMIT 3
        '''
        
        # Build query parameters
        search_tags = [f"%{purpose}%", "%Portable%", "%High-End%"]
        params = [budget] + search_tags + [f"%{purpose}%", "%Portable%"]
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "I couldn't find any laptops matching your requirements. Could you please adjust your criteria?"
            
        session['recommendation_links'] = {}
        
        # Construct the recommendation response
        response = ["Based on your requirements, here are some recommendations:"]
        for i, (model, price, weight, gpu, ram, tags, link) in enumerate(results, 1):
            session['recommendation_links'][str(i)] = link
            response.append(
                f"\n• {model}\n"
                f"  Price: ${price:,.2f}\n"
                f"  Weight: {weight}kg\n"
                f"  GPU: {gpu}\n"
                f"  RAM: {ram}GB\n"
                f"  Features: {tags}"
            )
            
        # Add follow-up suggestions
        response.append("\nWould you like to know more about any of these laptops? "
                       "Or shall we look for different options?"
                       "Or else type 1, 2, or 3 to select a laptop and get the purchase link.")
        
        return "\n".join(response)
        
    except Exception as e:
        print(f"Fallback system error: {e}")
        return ("I apologize, but I'm having trouble accessing our recommendation system. "
                "Please try again later or contact support.")

# ------------------------------------------------------------

# Helper function to analyze user preferences
def analyze_user_preferences(user_input, conversation_history):
    preferences = {
        'purpose': 'General',
        'budget': 2000,
        'portability': None,
        'brand': None
    }
    
    # Analyze current input and conversation history
    all_text = user_input.lower() + ' ' + ' '.join(
        msg['content'].lower() for msg in conversation_history 
        if msg['role'] == 'user'
    )
    
    # Usage analysis
    if any(word in all_text for word in ['game', 'gaming', 'play']):
        preferences['purpose'] = 'Gaming'
    elif any(word in all_text for word in ['work', 'business', 'office']):
        preferences['purpose'] = 'Business'
    elif any(word in all_text for word in ['study', 'student', 'school']):
        preferences['purpose'] = 'Student'
    
    # Budget analysis
    budget_mentions = re.findall(r'\$?(\d+)', all_text)
    if budget_mentions:
        potential_budget = int(budget_mentions[-1])
        if 100 <= potential_budget <= 5000: # Reasonable budget range
            preferences['budget'] = potential_budget
    
    # Portability analysis
    if any(word in all_text for word in ['portable', 'light', 'carry']):
        preferences['portability'] = 'Portable'
    elif any(word in all_text for word in ['desktop', 'replacement', 'powerful']):
        preferences['portability'] = 'Desktop Replacement'
    
    # Brand preference
    brands = ['dell', 'hp', 'lenovo', 'asus', 'acer', 'apple']
    for brand in brands:
        if brand in all_text:
            preferences['brand'] = brand.title()
            break
    
    return preferences

# ------------------------------------------------------------

@app.route('/start', methods=['GET'])
def start_conversation():
    print("Starting chatbot session...") # Debug log
    session['conversation'] = []
    session['step'] = 0
    
    # Reset conversation state
    session['conversation_state'] = {
        'step': 0,
        'purpose': None,
        'budget': None,
        'portability': None,
        'brand': None
    }
    
    try:
        print("Attempting to connect to OpenAI API...") # Debug log
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Ensure the correct model name is used
            messages=[
                {"role": "system", "content": "You are an expert laptop advisor. Start the conversation by warmly greeting the user and asking about their laptop needs. Be concise but friendly."},
                {"role": "user", "content": "Start the conversation"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        first_message = completion.choices[0].message.content
        print(f"API Response received: {first_message}") # Debug log
        
        session['conversation'].append({
            'role': 'assistant', 
            'content': first_message
        })
        
        return jsonify({'message': first_message})
        
    except Exception as e:
        print(f"Error in start_conversation: {e}") # Error log
        fallback_message = "Hi! I'm here to help you find the perfect laptop. What will you primarily use it for?"
        return jsonify({'message': fallback_message})

# ------------------------------------------------------------

@app.route('/')
def home():
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('index.html')

# ------------------------------------------------------------

@app.route('/chat')
def chat():
    if 'user_id' not in session:
        return redirect(url_for('home'))
    return render_template('index.html')

# ------------------------------------------------------------

def generate_recommendations():
    try:
        # Analyze key information from the conversation history
        conversation = session.get('conversation', [])
        user_inputs = [msg['content'] for msg in conversation if msg['role'] == 'user']
        
        # Use NLP or keyword matching to extract key details
        purpose, budget, preferences = analyze_conversation(conversation)
        
        # Query the database
        conn = sqlite3.connect(laptop_advisor_db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT model, price, weight, gpu, ram, tags
            FROM laptops
            WHERE price <= ? 
            AND tags LIKE ?
            ORDER BY price DESC
            LIMIT 3
        '''
        
        cursor.execute(query, (budget, f"%{purpose}%"))
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "I couldn't find any laptops matching your specific requirements. Let me adjust the search criteria."
        
        # Format recommendation results
        recommendations = ["Based on our conversation, here are my top recommendations:"]
        for model, price, weight, gpu, ram, tags in results:
            recommendations.append(
                f"\n• {model}\n"
                f"  Price: ${price:,.2f}\n"
                f"  Weight: {weight}kg\n"
                f"  GPU: {gpu}\n"
                f"  RAM: {ram}GB\n"
                f"  Features: {tags}"
            )
        
        return "\n".join(recommendations)
    
    except Exception as e:
        print(f"Error in recommendations: {e}")
        return "I apologize, but I'm having trouble generating recommendations right now."

# ------------------------------------------------------------

def analyze_conversation(conversation):
    # Implement conversation analysis logic here
    # Return extracted purpose, budget, and preferences
    # This is a simplified example
    purpose = "General"
    budget = 1500
    preferences = []
    
    for msg in conversation:
        content = msg['content'].lower()
        # Add more complex analysis logic
        if "gaming" in content:
            purpose = "Gaming"
        elif "business" in content or "work" in content:
            purpose = "Business"
        # Add more analysis rules...
    
    return purpose, budget, preferences

# ------------------------------------------------------------

# Convert an image file to base64 encoding
def encode_image_to_base64(image_file):
    try:
        # Read the uploaded file
        image = Image.open(image_file)
        
        # Resize the image to ensure it does not exceed API limits
        max_size = (2000, 2000) # Maximum size supported by GPT-4V
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Image encoding error: {e}")
        return None

# ------------------------------------------------------------

def analyze_laptop_image(image_base64, user_message=""):
    try:
        print("Starting laptop analysis with GPT-4o...")
        
        # Optimized system prompt
        system_prompt = """You are a helpful laptop advisor with expertise in model identification and technical analysis. 

        First, identify the laptop model by analyzing:
        • Brand and logo design
        • Series-specific design elements (gaming, business, consumer, etc.)
        • Model indicators and unique features
        • Generation-specific characteristics

        Then, as a helpful laptop advisor, analyze:

        1. Technical Features:
        - Display quality and specifications
        - Performance capabilities
        - Build quality and design
        - Cooling system
        - Available ports

        2. User Experience:
        - Keyboard and touchpad
        - Audio system
        - Portability
        - Special features

        3. Recommendations:
        - Best use cases
        - Notable strengths
        - Things to consider
        - User tips

        Focus on being helpful and informative, providing practical insights for users interested in this laptop."""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please identify this laptop model and provide helpful insights about its features and capabilities."
                    }
                ]
            }
        ]

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I couldn't analyze the laptop image. Please try again."

# ------------------------------------------------------------

@app.route('/chat', methods=['POST'])
def chatbot():
    try:
        print("Received chat request") # Debug log
        
        user_id = session['user_id']
        user_message = request.form.get('message', '').strip()
        image_file = request.files.get('image')
        
    
        print(f"Message: {user_message}") # Debug log
        print(f"Image file present: {image_file is not None}") # Debug log
        
        if 'conversation' not in session:
            session['conversation'] = []
            session['current_analysis'] = None
            session['recommendation_links'] = {}
        
        # Handle image upload
        if image_file:
            try:
                print("Processing image file...") # Debug log
                
                # Validate file type
                if not allowed_file(image_file.filename):
                    return jsonify({"error": "Invalid file type"}), 400
                
                # Encode image
                image_base64 = encode_image_to_base64(image_file)
                if not image_base64:
                    return jsonify({"error": "Failed to encode image"}), 400
                
                # Analyze the image
                ai_response = analyze_laptop_image(image_base64, user_message)
                
                print(f"AI Response: {ai_response[:100]}...") # Debug log
                
                # Save analysis result to session
                session['current_analysis'] = ai_response
                
                # Update conversation history
                session['conversation'].extend([
                    {"role": "user", "content": f"[IMAGE_ANALYSIS_REQUEST] {user_message}"},
                    {"role": "assistant", "content": ai_response}
                ])
                
                # Save chat history
                save_history(user_id, session['conversation'])
                
                return jsonify({"reply": ai_response})
                
            except Exception as e:
                print(f"Image processing error: {str(e)}") # Error log
                return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
        
        # Handle numeric input before calling parse_user_input
        if user_message in ['1', '2', '3']:
            if 'recommendation_links' in session:
                selected = user_message
                link = session['recommendation_links'].get(selected)
                if link:
                    # Return a Markdown-formatted link
                    return f"🔗 Purchase Link for Option {selected}:\n[Click Here]({link})"
                else:
                    return "⚠️ Sorry, the purchase link is currently unavailable..."
            else:
                return "ℹ️ Please start a recommendation session first."
                
        # Handle text messages
        if user_message:
            response = parse_user_input(user_message, session['conversation'])
            
            # Update conversation history
            session['conversation'].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ])
            
            save_history(user_id, session['conversation'])
            return jsonify({"reply": response})
            
        return jsonify({"error": "No message or image provided"}), 400
        
    except Exception as e:
        print(f"General error in chatbot route: {str(e)}") # Error log
        return jsonify({"error": "An unexpected error occurred"}), 500

# ------------------------------------------------------------

# Add configuration to limit image upload size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max-limit

# ------------------------------------------------------------

# Add a check for allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# ------------------------------------------------------------

# Add error handling
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large. Maximum file size is 16MB."
    }), 413

# ------------------------------------------------------------

# Log error reports
@app.route('/report_error', methods=['POST'])
def report_error():
    user_id = session['user_id']
    error_details = request.json.get('error')
    log_error_to_db(user_id, error_details)  
    return jsonify({"status": "success"})

# ------------------------------------------------------------

# Store user-reported errors in the database
def log_error_to_db(user_id, error_details):
    try:
        # Connect to the database
        conn = sqlite3.connect('error_logs.db')
        cursor = conn.cursor()

        # Create the error log table (if it does not exist)
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                error_details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert error record
        cursor.execute('''
            INSERT INTO errors (user_id, error_details)
            VALUES (?, ?)
        ''', (user_id, error_details))

        # Commit changes and close connection
        conn.commit()
        conn.close()

    except Exception as e:
        # If an exception occurs, log it to the console (or a log file)
        print(f"Error logging to database: {e}")

# ------------------------------------------------------------

def init_db():
    conn = sqlite3.connect('laptop_advisor.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create chat history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        conversation TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# ------------------------------------------------------------

def save_history(user_id, conversation):
    conn = sqlite3.connect(chat_history_db_path)
    cursor = conn.cursor()
    
    # Convert conversation content into a string
    conversation_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in conversation
    )
    
    # Insert record with user_id
    cursor.execute(
        'INSERT INTO history (user_id, conversation) VALUES (?, ?)', 
        (user_id, conversation_text)
    )
    conn.commit()   
    conn.close()

# ------------------------------------------------------------

@app.route('/get_history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    user_id = session['user_id'] # Retrieve user_id from session
    conn = sqlite3.connect('chat_history.db', uri=True)
    conn.execute(f"PRAGMA key = {app.secret_key}")
    cursor = conn.cursor()
    
    # Query only the records belonging to the current user
    cursor.execute(
        'SELECT conversation, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC', 
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return jsonify({'error': 'No history found for this user.'})
    
    # Format history records
    history = [
        {
            'conversation': row[0],
            'timestamp': row[1]
        }
        for row in rows
    ]
    
    return jsonify(history)

# ------------------------------------------------------------

@app.route('/debug_session')
def debug_session():
    return jsonify(dict(session))

# ------------------------------------------------------------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
        
    try:
        conn = sqlite3.connect('laptop_advisor.db')
        cursor = conn.cursor()
        
        # Check if the username already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return jsonify({'error': 'Username already exists'}), 400
            
        # Create a new user
        hashed_password = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
            (username, hashed_password, email)
        )
        conn.commit()
        
        return jsonify({'message': 'Registration successful'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

# ------------------------------------------------------------

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    try:
        conn = sqlite3.connect('laptop_advisor.db')
        cursor = conn.cursor()
        
        # Query user
        cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = str(user[0]) # Convert user_id to a string and store it in the session
            session['username'] = username
            return jsonify({'message': 'Login successful'})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    finally:
        conn.close()

# ------------------------------------------------------------

@app.route('/logout', methods=['POST'])
def logout():
    try:
        print("Logout route accessed") # Debug log
        print("Session before clear:", dict(session)) # Debug log
        session.clear()
        print("Session after clear:", dict(session)) # Debug log
        return jsonify({'success': True, 'message': 'Logout successful'})
    except Exception as e:
        print(f"Logout error: {e}") # Debug log
        return jsonify({'success': False, 'message': str(e)}), 500

# ------------------------------------------------------------

# Ensure the route supports OPTIONS requests (handle CORS)
@app.route('/logout', methods=['POST', 'OPTIONS'])
def handle_logout():
    if request.method == 'OPTIONS':
        return '', 204
    return logout()

# ------------------------------------------------------------

# Initialize the database when the program starts
init_db()

# ------------------------------------------------------------

# Conversation state loss
@app.before_request
def before_request():
    if 'conversation_state' not in session:
        print("Initializing conversation state") # Debug log
        session['conversation_state'] = {
            'step': 0,
            'purpose': None,
            'budget': None,
            'portability': None,
            'brand': None
        }

# ------------------------------------------------------------

# Add an error handler
@app.errorhandler(500)
def handle_500_error(e):
    print(f"500 error occurred: {str(e)}") # Error log
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again.",
        "details": str(e)
    }), 500

# ------------------------------------------------------------

# Add a simple health check endpoint
@app.route('/health')
def health_check():
    try:
        # Test database connection
        conn = sqlite3.connect(laptop_advisor_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "session": "active" if 'conversation_state' in session else "inactive"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# ------------------------------------------------------------

# Add a test function
@app.route('/test_api', methods=['GET'])
def test_api():
    try:
        # Test API connection
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        print("API test response:", completion.choices[0].message.content) # Debug log
        return jsonify({"status": "success", "message": "API connection successful"})
        
    except Exception as e:
        print(f"API test error: {e}") # Error log
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------------------------------------

# Initialize the laptop database
def init_laptop_db():
    conn = sqlite3.connect(laptop_advisor_db_path)
    cursor = conn.cursor()
    
    # Create the laptops table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS laptops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT NOT NULL,
        price REAL NOT NULL,
        weight REAL,
        gpu TEXT,
        ram INTEGER,
        tags TEXT,
        cpu TEXT,
        storage TEXT,
        display TEXT,
        battery TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# ------------------------------------------------------------

# Check database connection and content
def check_database():
    try:
        conn = sqlite3.connect(laptop_advisor_db_path)
        cursor = conn.cursor()
        
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='laptops'")
        if not cursor.fetchone():
            print("Error: laptops table does not exist")
            return False
            
        # Check the number of records
        cursor.execute("SELECT COUNT(*) FROM laptops")
        count = cursor.fetchone()[0]
        print(f"Database contains {count} laptop records")
        
        # Check sample data
        cursor.execute("SELECT * FROM laptops LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print("Sample data:", sample)
        
        return True
    
    except Exception as e:
        print(f"Database check error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# ------------------------------------------------------------

# Check database when the application starts
if __name__ == '__main__':
    init_laptop_db()
    if not check_database():
        print("Warning: Database check failed!")
    print("Server starting...")
    app.run(ssl_context='adhoc') # Run with HTTPS for testing
    app.run(host='127.0.0.1', port=3000, debug=True)

print(os.path.exists(laptop_advisor_db_path)) # Should return True