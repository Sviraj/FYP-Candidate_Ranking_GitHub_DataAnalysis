from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.metrics import mean_absolute_error
import os
from PyPDF2 import PdfReader
import re
from urlextract import URLExtract
import csv
import requests
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from skopt import BayesSearchCV
from skopt.space import Real, Integer

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print('That 28 line executed')

# Download required NLTK data files
nltk.download('punkt')
print('That 22 line executed')

# Load the CSV file into a DataFrame
# df = pd.read_csv('labeledDataSet.csv')
df = pd.read_csv('Formatted_Data1.csv')
print('That 65 line executed now')

# Separate features (X) and target variable (y)
X = df.drop(['username', 'industry_score'], axis=1)
y = df['industry_score']
print('That 142 line executed')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define Hyperparameter Grid
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [10, 20, 30, 40, 50, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# # Initialize and train a Random Forest Regressor
# rf_regressor = RandomForestRegressor(random_state=42)    #n_estimators=100, random_state=42
# #rf_regressor.fit(X_train, y_train)
# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# # Fit GridSearchCV
# grid_search.fit(X_train, y_train)
# # Get best parameters
# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)
# # Train final model with best parameters
# final_model = RandomForestRegressor(**best_params, random_state=42)
# final_model.fit(X_train, y_train)
# # Check if the model is fitted
# def is_model_fitted(model):
#     try:
#         check_is_fitted(model)
#         return True
#     except:
#         return False
# print(is_model_fitted(final_model))  # True, because the model is now fitted
# # Evaluate the model
# if is_model_fitted(final_model):
#     y_pred_rf = final_model.predict(X_test)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     print(f"Random Forest Regressor MAE: {mae_rf}")
#     print(f"Random Forest Regressor R2 Score: {r2_rf}")
# else:
#     print("Model is not fitted yet.")


# Define Hyperparameter Space
param_space_rf = {
    'n_estimators': Integer(100, 300),
    'max_depth': Integer(10, 30),
    'min_samples_split': Integer(2, 11),
    'min_samples_leaf': Integer(1, 11)
}

# Initialize RandomForestRegressor
rf_rf = RandomForestRegressor(random_state=42)

# Initialize and Fit BayesSearchCV
bayes_search = BayesSearchCV(estimator=rf_rf, search_spaces=param_space_rf, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
bayes_search.fit(X_train, y_train)

# Get best parameters and train final model
best_params = bayes_search.best_params_
print("Best parameters found: ", best_params)

# Use the best estimator from BayesSearchCV directly
best_rf_model = bayes_search.best_estimator_

# Check if the model is fitted
def is_model_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except:
        return False

print(is_model_fitted(best_rf_model))  # True, because the model is now fitted
# Evaluate the model
if is_model_fitted(best_rf_model):
    y_pred_rf = best_rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest Regressor MAE: {mae_rf}")
    print(f"Random Forest Regressor R2 Score: {r2_rf}")
else:
    print("RF_Model is not fitted yet.")


print('That 154 line executed')
print(X_test)
print(y_test)

# Evaluate the Random Forest Regressor
#y_pred_rf = rf_regressor.predict(X_test)
# y_pred_rf = final_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)
#
# print(f"Random Forest Regressor MAE: {mae_rf}")
# print(f"Random Forest Regressor R2 Score: {r2_rf}")

# Define the parameter space
param_space_dt = {
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 10)
}

# Initialize and train a Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Initialize BayesSearchCV
bayes_search = BayesSearchCV(estimator=dt_regressor, search_spaces=param_space_dt, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)
bayes_search.fit(X_train, y_train)

# Get the best parameters
best_params = bayes_search.best_params_
print("Best parameters found: ", best_params)

# Evaluate the model
best_dt_model = bayes_search.best_estimator_

print(is_model_fitted(best_dt_model))  # True, because the model is now fitted

if is_model_fitted(best_dt_model):
    # Evaluate the model
    y_pred_dt = best_dt_model.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    print(f"Decision Tree Regressor MAE: {mae_dt}")
    print(f"Decision Tree Regressor R2 Score: {r2_dt}")
else:
    print("DT_Model is not fitted yet.")

# Function to extract URLs from text using urlextract
def extract_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    print('Extract Urls Complete:', urls)
    return urls


def extract_github_urls(text):
    # Regular expression pattern to match GitHub URLs
    # pattern = r'(https?://github\.com/[^\s]+)'
    # pattern = r'\b(?:https?://)?github\.com/\w+'
    # pattern = r'\b(?:https?://github\.com/|github\.com/)\w+'
    pattern = r'\b(?:https?://|http://|www\.)?github\.com/\w+'
    github_urls = re.findall(pattern, text)
    print('Extract Github url complete:', github_urls)
    return github_urls


# def extract_languages_from_cv(text):
#     programming_languages = ['Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'C#', 'Swift', 'Go', 'R', 'Kotlin']
#     cv_languages = []
#     tokens = word_tokenize(text)
#     for token in tokens:
#         if token in programming_languages:
#             cv_languages.append(token)
#     return cv_languages

def extract_languages_from_cv(text):
    programming_languages = {'Python', 'Java', 'C++', 'JavaScript', 'Ruby', 'PHP', 'C#', 'Swift', 'Go', 'R', 'Kotlin'}
    cv_languages = set()
    tokens = word_tokenize(text)
    for token in tokens:
        if token in programming_languages:
            cv_languages.add(token)
    return list(cv_languages)


def check_similarity(cv_languages, github_languages):
    print('This similarity checking function is working')
    similarity_score = 0
    for lang in cv_languages:
        if lang in github_languages:  # and github_languages[lang] > 50
            similarity_score += 1
    return similarity_score


def scrape_github_contributions(username):
    # Setup Chrome WebDriver
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run headless Chrome to avoid opening a browser window

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(f'https://github.com/{username}')
    driver.implicitly_wait(2)

    contributions_data = {}
    try:
        # Find the contributions in the last year element
        contributions_element = driver.find_element(By.XPATH, '//h2[contains(@class, "f4 text-normal mb-2")]')
        contributions_text = contributions_element.text.strip()

        # Extract the number of contributions from the text
        contributions_number = contributions_text.split()[0]

        contributions_data = {
            'username': username,
            'contributions_last_year': int(contributions_number)
        }
    except Exception as e:
        contributions_data = {"error": str(e)}
    finally:
        driver.quit()

    print('Github Scraping Complete:', contributions_data)
    return contributions_data


def check_passion(text):
    print('text', text)
    # Check if bio is None or empty
    if text is None or not text.strip():
        return "no bio appears in profile"
    # Sample bio text
    bio = text
    # Define keywords for frontend and backend
    frontend_keywords = {'frontend', 'front-end', 'UI', 'user interface', 'UX', 'user experience', 'HTML', 'CSS',
                         'JavaScript'}
    backend_keywords = {'backend', 'back-end', 'server', 'database', 'API', 'Python', 'Java', 'Node.js'}
    # Tokenize the bio text
    tokens = word_tokenize(bio.lower())
    # Check for the presence of frontend or backend keywords
    frontend_count = sum(1 for token in tokens if token in frontend_keywords)
    backend_count = sum(1 for token in tokens if token in backend_keywords)
    # Determine the user's preference
    if frontend_count > backend_count:
        preference = 'frontend development'
    elif backend_count > frontend_count:
        preference = 'backend development'
    else:
        preference = 'unspecified or equal preference for frontend and backend development'

    print('Check Passion Completed:', preference)
    return preference


def check_more_toward_stack(data):
    print('Check More Toward Stack Complete')
    new_data = {key: value for key, value in data.items() if
                key not in ['public_repos', 'public_gists', 'followers', 'following', 'repos_count', 'contribution']}
    # Print the new dictionary
    # print(new_data)
    # Define sets of frontend and backend technologies
    frontend_technologies = {'JavaScript', 'HTML', 'CSS', 'TypeScript', 'Dart'}
    backend_technologies = {'Java', 'Python', 'C++', 'C#', 'PHP', 'Go', 'Kotlin', 'C'}

    # Initialize sum variables
    frontend_sum = 0.0
    backend_sum = 0.0

    # Calculate the sum of values for frontend and backend technologies
    for tech, value in new_data.items():
        value = float(value)  # Convert the value from string to float
        if tech in frontend_technologies:
            frontend_sum += value
        elif tech in backend_technologies:
            backend_sum += value

    # Determine which development area the data is more oriented towards
    if frontend_sum == 0 and backend_sum == 0:
        return "Not Data to Predict Stack"
    elif frontend_sum > backend_sum:
        return "More oriented towards Frontend Development"
    elif backend_sum > frontend_sum:
        return "More oriented towards Backend Development"
    else:
        return "Equally oriented towards Frontend and Backend Development"


def predict(data):
    if data:
        user_bio = data['bio']
        data.pop('username', None)
        data.pop('bio', None)
        # data.pop('contribution', None)
        suitable_stack = check_more_toward_stack(data)
        # Data is sent from frontend, process it and make predictions
        # data = request.json  # Assuming JSON data is sent from frontend
        new_data = pd.DataFrame([data])
    else:
        # No data sent, use default data for testing purposes
        print("This is dummy data")
        new_data = pd.DataFrame({
            'public_repos': [79],
            'public_gists': [72],
            'followers': [42],
            'following': [3],
            'repos_count': [30],
            'contribution': [100],
            'JavaScript': [44],
            'Java': [1],
            'HTML': [37],
            'CSS': [13],
            'C': [0],
            'Python': [11],
            'C++': [0],
            'TypeScript': [56],
            'C#': [5],
            'Dart': [0],
            'Kotlin': [0],
            'PHP': [35],
            'Go': [13]
        })

    # Make predictions using the models
    predicted_score_dt = best_dt_model.predict(new_data)
    #predicted_score_rf = rf_regressor.predict(new_data)
    predicted_score_rf = best_rf_model.predict(new_data)

    # print('Predicted Score (Decision Tree):', predicted_score_dt[0])
    # print('Predicted Score (Random Forest):', predicted_score_rf[0])

    # Render index.html template with predicted scores
    # return render_template('index.html', predicted_score_dt=predicted_score_dt[0], predicted_score_rf=predicted_score_rf[0])
    predicted_score_dt = predicted_score_dt[0]
    predicted_score_rf = predicted_score_rf[0]
    user_preference = check_passion(user_bio)

    print('Predict Function Completed:', predicted_score_dt, predicted_score_rf, user_preference, suitable_stack)
    return predicted_score_dt, predicted_score_rf, user_preference, suitable_stack


def get_user_data(username, token):
    url = f'https://api.github.com/users/{username}'
    headers = {'Authorization': f'token {token}'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print('GET request successful!')
        else:
            print(f'GET request failed with status code {response.status_code}. User Profile can not found')
        # print('that line work')
        # print(response.json())
        profile_data = response.json()
        # print(profile_data['bio'])
        # for key, value in profile_data.items():
        # print(f"{key}: {value}")
    except Exception as e1:
        print(f"An unexpected error occured when fetching profile data : {e1}")

    repos_data_url = profile_data['repos_url']
    try:
        repos_data_response = requests.get(repos_data_url, headers=headers)
        repos_data = repos_data_response.json()
    except Exception as e2:
        print(f"An unexpected error occured when fetching repos data : {e2}")

    repos_count = len(repos_data)
    print('Repos Count:', repos_count)

    cumulative_data = {}
    for obj in repos_data:
        try:
            lang_data_url = obj['languages_url']
            tech_data_response = requests.get(lang_data_url, headers=headers)
            tech_data = tech_data_response.json()
        except Exception as e3:
            print(f"An unexpected error occured when fetching profile data : {e3}")

        total_sum = sum(tech_data.values())
        percentage_objects = [{'key': key, 'percentage': (value / total_sum) * 100} for key, value in tech_data.items()]

        for objet in percentage_objects:
            key = objet['key']
            percentage = objet['percentage']
            if key in cumulative_data:
                cumulative_data[key]['sum'] += percentage
                cumulative_data[key]['count'] += 1
                cumulative_data[key]['average'] = cumulative_data[key]['sum'] / cumulative_data[key]['count']
            else:
                cumulative_data[key] = {'sum': percentage, 'count': 1, 'average': percentage}

    average_list = [(language, format(details['average'], '.2f')) for language, details in cumulative_data.items()]
    language_average_dict = dict(average_list)

    try:
        git_contribution = scrape_github_contributions(username)
        print('git contribution:', git_contribution)
    except Exception as e4:
        print("An unexpected error occured when web scraping : {e4}")

    my_dict = {
        'username': profile_data['login'],
        'bio': profile_data['bio'],
        'public_repos': profile_data['public_repos'],
        'public_gists': profile_data['public_gists'],
        'followers': profile_data['followers'],
        'following': profile_data['following'],
        'repos_count': repos_count,
        'contribution': git_contribution['contributions_last_year'],
        'JavaScript': language_average_dict.get('JavaScript', 0),
        'Java': language_average_dict.get('Java', 0),
        'HTML': language_average_dict.get('HTML', 0),
        'CSS': language_average_dict.get('CSS', 0),
        'C': language_average_dict.get('C', 0),
        'Python': language_average_dict.get('Python', 0),
        'C++': language_average_dict.get('C++', 0),
        'TypeScript': language_average_dict.get('TypeScript', 0),
        'C#': language_average_dict.get('C#', 0),
        'Dart': language_average_dict.get('Dart', 0),
        'Kotlin': language_average_dict.get('Kotlin', 0),
        'PHP': language_average_dict.get('PHP', 0),
        'Go': language_average_dict.get('Go', 0),
    }
    print('my dict', my_dict)
    return my_dict


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and file.filename.endswith('.pdf'):
            # Save the uploaded PDF file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Extract text content from the PDF
            pdf_text = ""
            with open(filename, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

            # print(pdf_text)
            # Extract URLs from the text
            urls = extract_urls(pdf_text)
            # print(f"urls Extracted:{urls}")
            github_urls = extract_github_urls(pdf_text)
            print(f"github url :{github_urls}")
            extracted_Lang_cv = extract_languages_from_cv(pdf_text)
            print('extracted language cv', extracted_Lang_cv)
            # Check if there are any GitHub URLs to process
            if not github_urls:
                return "Cannot extract GitHub URL"

            # Initialize an empty list to store extracted usernames
            usernames = []

            # Define the regex pattern outside the loop for better performance
            pattern = r'github\.com/(\w+)'

            for url in github_urls:
                match = re.search(pattern, url)
                if match:
                    username = match.group(1)
                    usernames.append(username)
                else:
                    usernames.append("Username not found")

            # github_token = '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'
            github_token = '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'

            for username in usernames:
                print(username)
                user_data = get_user_data(username, github_token)

            values = predict(user_data)
            # Remove specific keys in one line
            [user_data.pop(key, None) for key in
             ['username', 'bio', 'public_repos', 'public_gists', 'followers', 'following', 'repos_count',
              'contribution']]
            print('updated_user_data', user_data)
            similarity_score = check_similarity(extracted_Lang_cv, user_data)
            print('similarity_score', similarity_score)
            # print(values[0])
            # print('suitable stack', values[3])
            print('Succefull Complete Last Function:', username, values)
            return render_template('urls.html', username=username, predicted_score_dt=values[0],
                                   predicted_score_rf=values[1], user_preference=values[2], suitable_stack=values[3])
        else:
            return "Invalid file format. Please upload a PDF file."
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
