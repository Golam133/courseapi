from flask import Flask, jsonify, request
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import random

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key

# Step 1: Set up the MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['learning_platform']
courses_collection = db['courses']
interactions_collection = db['user_interactions']

# Step 2: Define route to load CSV data into MongoDB
@app.route('/load_data', methods=['POST'])
def load_data():
    course_file = r'C:\Users\User\Desktop\400\putaa - Sheet1 (1).csv'
    interaction_file = r'C:\Users\User\Desktop\400\simulated_user_interactions.csv'
    combined_df = pd.read_csv(course_file)
    user_interaction_df = pd.read_csv(interaction_file)
    combined_df.rename(columns={'Course ID': 'course_id'}, inplace=True)
    courses_data = combined_df.to_dict(orient='records')
    interactions_data = user_interaction_df.to_dict(orient='records')
    courses_collection.insert_many(courses_data)
    interactions_collection.insert_many(interactions_data)
    return jsonify({"message": "Data successfully inserted into MongoDB!"}), 201

# Step 3: Define route to retrieve all courses
@app.route('/courses', methods=['GET'])
def get_courses():
    courses = list(courses_collection.find({}, {"_id": 0}))
    return jsonify(courses)

# Generate quiz questions based on a course from DB
@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    course_name = data.get('course_name')
    
    if not course_name:
        courses = list(courses_collection.find({}, {'_id': 0, 'Title': 1}))
        course_name = random.choice(courses)['Title']
    
    prompt = f"Generate 3 multiple-choice questions about {course_name}. Include 4 answer options and specify the correct answer."
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    
    quiz = response.choices[0].text.strip()
    return jsonify({'quiz': quiz})

# Evaluate quiz answers and provide feedback
@app.route('/evaluate_quiz', methods=['POST'])
def evaluate_quiz():
    data = request.json
    user_answers = data.get('user_answers')
    correct_answers = data.get('correct_answers')
    
    feedback = []
    for i, (user_ans, correct_ans) in enumerate(zip(user_answers, correct_answers)):
        if user_ans == correct_ans:
            feedback.append(f"Question {i+1}: Correct!")
        else:
            feedback.append(f"Question {i+1}: Incorrect. The correct answer was {correct_ans}. Review this topic.")
    
    return jsonify({'feedback': feedback})

# Load the data from MongoDB into DataFrames
course_data = list(courses_collection.find({}, {"_id": 0}))
interaction_data = list(interactions_collection.find({}, {"_id": 0}))
combined_df = pd.DataFrame(course_data)
user_interaction_df = pd.DataFrame(interaction_data)

# Ensure combined_df and user_interaction_df have necessary columns
required_columns = ['Title', 'course_id', 'Categories']
if not all(col in combined_df.columns for col in required_columns):
    def assign_category(title):
        title = title.lower()
        if 'web' in title:
            return 'Web Development'
        elif 'machine learning' in title or 'ml' in title:
            return 'Machine Learning'
        elif 'artificial intelligence' in title or 'ai' in title:
            return 'Artificial Intelligence'
        elif 'python' in title:
            return 'Python'
        elif 'java' in title:
            return 'Java'
        elif 'data structure' in title:
            return 'Data Structure'
        elif 'network' in title:
            return 'Computer Network'
        elif 'sql' in title:
            return 'SQL'
        else:
            return 'Other'

    combined_df['Categories'] = combined_df['Title'].apply(assign_category)

if 'course_id' not in user_interaction_df.columns:
    raise ValueError("Missing 'course_id' in user_interaction_df")

# Content-Based Filtering
def content_based_filtering_by_category(category):
    filtered_df = combined_df[combined_df['Categories'] == category]
    if filtered_df.empty:
        return pd.DataFrame()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['Title'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    similar_indices = cosine_sim[0].argsort()[-10:][::-1]
    return filtered_df.iloc[similar_indices]

# Collaborative Filtering
def collaborative_filtering(user_id):
    user_interactions = user_interaction_df[user_interaction_df['user_id'] == user_id]['course_id'].tolist()
    similar_users = user_interaction_df[user_interaction_df['course_id'].isin(user_interactions) & 
                                        (user_interaction_df['user_id'] != user_id)]['user_id'].unique()
    
    recommended_courses = user_interaction_df[(user_interaction_df['user_id'].isin(similar_users)) & 
                                              (~user_interaction_df['course_id'].isin(user_interactions))]['course_id'].unique()
    return recommended_courses

# Dynamic Weight Assignment Based on User Interactions
def get_dynamic_weights(user_id):
    num_interactions = len(user_interaction_df[user_interaction_df['user_id'] == user_id])
    if num_interactions < 5:
        return 0.8, 0.2
    elif 5 <= num_interactions <= 20:
        return 0.6, 0.4
    else:
        return 0.3, 0.7

# Hybrid Recommendation System
@app.route('/recommend', methods=['GET'])
def weighted_hybrid_recommendation():
    user_id = int(request.args.get('user_id'))
    category = request.args.get('category')

    if user_interaction_df[user_interaction_df['user_id'] == user_id].empty:
        content_recommendations_df = content_based_filtering_by_category(category)
        collaborative_recommendations = collaborative_filtering(user_id)

        num_content_recommendations = int(len(content_recommendations_df) * 0.8)
        num_collaborative_recommendations = int(len(collaborative_recommendations) * 0.2)

        content_weighted = content_recommendations_df.sample(n=num_content_recommendations, replace=True)
        collaborative_weighted = combined_df[combined_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

        hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

        if hybrid_recommendations.empty:
            return jsonify({"message": "No recommendations found"}), 404

        return jsonify(hybrid_recommendations.to_dict(orient='records'))

    else:
        content_weight, collaborative_weight = get_dynamic_weights(user_id)
        content_recommendations_df = content_based_filtering_by_category(category)
        collaborative_recommendations = collaborative_filtering(user_id)

        num_content_recommendations = int(len(content_recommendations_df) * content_weight)
        num_collaborative_recommendations = int(len(collaborative_recommendations) * collaborative_weight)

        content_weighted = content_recommendations_df.sample(n=num_content_recommendations, replace=True)
        collaborative_weighted = combined_df[combined_df['course_id'].isin(collaborative_recommendations)].sample(n=num_collaborative_recommendations, replace=True)

        hybrid_recommendations = pd.concat([content_weighted, collaborative_weighted]).drop_duplicates(subset='course_id', keep='first')

        if hybrid_recommendations.empty:
            return jsonify({"message": "No recommendations found"}), 404

        return jsonify(hybrid_recommendations.to_dict(orient='records'))

# Route to list available categories
@app.route('/categories', methods=['GET'])
def list_categories():
    categories = combined_df['Categories'].unique()
    return jsonify(categories.tolist())

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
