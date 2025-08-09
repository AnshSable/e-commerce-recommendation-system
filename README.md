# e-commerce-recommendation-system

## 📌 Overview
This project is a **Recommendation System** built using:
- **Collaborative Filtering** (Item-based using Surprise library)
- **Cold Start Problem Handling** for:
  - New Users → Popular products
  - New Products → Similar item-based recommendations (fallback: popular products)
- **Streamlit Web App** for an interactive interface

The model is trained on the **Amazon Electronics Ratings Dataset** and deployed using Streamlit.

---

## 🚀 Features
- **Personalized Recommendations** (for existing users)
- **Cold Start Handling**:
  - New users → Top popular products
  - New products → Similar items or popular items
- **Fast Predictions** using a pre-trained `.pkl` model
- **Interactive Web App** with Streamlit
- **Dataset filtering & EDA** included

---

## 🗂 Project Structure
📦 project-folder
┣ 📜 app.py # Streamlit application
┣ 📜 item_based_cf_model.pkl # Trained Surprise model
┣ 📜 ratings_Electronics.csv # Dataset (sample for demo)
┣ 📜 requirements.txt # Dependencies
┣ 📜 README.md # Project documentation
┣ 📜 .gitignore # Ignore venv/pycache
┗ 📂 screenshots # Folder for app screenshots



---

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate      # For Mac/Linux
venv\Scripts\activate         # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🖥 Usage
User-based Recommendation:

Enter a user_id from the dataset

If user is new → Popular products are shown

Product-based Recommendation:

Enter a product_id from the dataset

If product is new → Popular products are shown

📷 Screenshots
🏠 Home Page

📊 User-based Recommendation

📦 Product-based Recommendation
