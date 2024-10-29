from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# Initialize the LLM and chains
api_key = "YOUR_API_KEY"
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)

# Create prompt templates
genre_prompt = PromptTemplate(
    input_variables=["user_input", "available_genres"],
    template="""
    Given the following user input: {user_input}
    
    And these available genres: {available_genres}
    
    Please suggest the most relevant genres from the available list that match the user's input.
    Return only the genre names separated by commas, without any additional explanation.
    If no genres match, return "No matching genres found".
    """
)

recommendation_prompt = PromptTemplate(
    input_variables=["user_input", "filtered_content"],
    template="""
    Based on the user's request: {user_input}
    
    And the following available content:
    {filtered_content}
    
    Please provide a natural, conversational response that includes:
    1. Relevant titles that match their interests
    2. Release years for each recommendation
    3. A brief explanation of why each title matches their request
    
    Focus on being helpful and informative while maintaining a natural tone.
    If no content matches, explain that no exact matches were found and suggest broadening the search.
    """
)

genre_chain = LLMChain(llm=llm, prompt=genre_prompt)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Load your DataFrame and genres_set
try:
    with open('genres_set.pkl', 'rb') as f:
        genres_set = pickle.load(f)
    df = pd.read_pickle('movies_df.pkl')

except FileNotFoundError:
    print("Error: Data files not found!")
    genres_set = set()
    df = pd.DataFrame()

def filter_content(df, genres):
    if not genres:
        return pd.DataFrame()
    
    mask = df['genres'].apply(
        lambda x: any(genre.strip() in [g.strip() for g in x.split(',')] for genre in genres)
    )
    filtered_df = df[mask]
    print(f"Filtered {len(filtered_df)} movies for genres: {genres}")
    return filtered_df

def format_content_for_prompt(filtered_df):
    if filtered_df.empty:
        return "No matching content found."
    
    content_list = [
        f"Title: {row['title']}\nGenres: {row['genres']}\nRelease Year: {row['releaseYear']}\nRating: {row['averageRating']} (from {row['numVotes']} votes)\n"
        for _, row in filtered_df.iterrows()
    ]
    formatted_content = "\n".join(content_list)
    print(f"Formatted content length: {len(formatted_content)}")
    return formatted_content

@app.route('/')
def home():
    return render_template('index.html', genres=sorted(list(genres_set)))

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.form.get('user_input', '')
        print(f"\nReceived user input: {user_input}")
        
        # Get suggested genres
        available_genres = sorted(list(genres_set))
        print(f"Available genres: {available_genres[:5]}...")
        
        genre_response = genre_chain.invoke({"user_input": user_input, "available_genres": ", ".join(available_genres)})
        suggested_genres = genre_response['text'].split(',')
        suggested_genres = [genre.strip() for genre in suggested_genres if genre.strip() in genres_set]
        print(f"Suggested genres: {suggested_genres}")
        
        # Filter and format content
        filtered_content = filter_content(df, suggested_genres)
        formatted_content = format_content_for_prompt(filtered_content)
        print(f"Formatted content sample: {formatted_content[:200]}...")
        
        # Get recommendations
        recommendation_response = recommendation_chain.invoke({
            "user_input": user_input,
            "filtered_content": formatted_content
        })
        
        recommendations = recommendation_response['text']
        print(f"Recommendation response sample: {recommendations[:200]}...")
        
        response_data = {
            'success': True,
            'recommendations': recommendations,
            'suggested_genres': suggested_genres
        }
        print("Sending response data to client")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)