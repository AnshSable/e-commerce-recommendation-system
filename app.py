# import streamlit as st
# import pandas as pd
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from collections import defaultdict

# # Load dataset
# @st.cache_data
# def load_data():
#     df = pd.read_csv("filtered_df.csv")
#     return df

# # Load Surprise model (Optional - or train here)
# @st.cache_resource
# def train_model(df):
#     reader = Reader(rating_scale=(1, 5))
#     data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
#     trainset = data.build_full_trainset()
#     model = SVD()
#     model.fit(trainset)

#     # Predict for all pairs (for demo)
#     testset = trainset.build_anti_testset()
#     predictions = model.test(testset)
#     return model, trainset, predictions

# def get_top_n(predictions, n=10):
#     top_n = defaultdict(list)
#     for uid, iid, true_r, est, _ in predictions:
#         top_n[uid].append((iid, est))
#     for uid, user_ratings in top_n.items():
#         user_ratings.sort(key=lambda x: x[1], reverse=True)
#         top_n[uid] = user_ratings[:n]
#     return top_n

# def get_popular_products(df, n=10):
#     product_stats = df.groupby('product_id').agg({
#         'rating': ['mean', 'count']
#     }).reset_index()
#     product_stats.columns = ['product_id', 'avg_rating', 'rating_count']
#     top_n = product_stats.sort_values(by='rating_count', ascending=False).head(n)
#     return top_n

# # Load Data and Model
# df = load_data()
# model, trainset, predictions = train_model(df)
# top_n_dict = get_top_n(predictions)

# # UI Layout
# st.title("🛒 E-commerce Recommendation System")
# st.markdown("Get personalized or popular product recommendations!")

# user_input = st.text_input("Enter your User ID:", "")

# if st.button("Get Recommendations"):
#     if user_input:
#         if user_input in trainset._raw2inner_id_users:
#             st.subheader(f"🎯 Personalized Recommendations for User: {user_input}")
#             user_recs = top_n_dict.get(user_input, [])
#             for i, (iid, rating) in enumerate(user_recs, 1):
#                 st.write(f"{i}. Product ID: {iid} | Predicted Rating: {rating:.2f}")
#         else:
#             st.subheader(" New User! Showing Popular Products")
#             top_pop = get_popular_products(df, n=10)
#             for i, row in top_pop.iterrows():
#                 st.write(f"{i+1}. Product ID: {row['product_id']} | Avg Rating: {row['avg_rating']:.2f} | Count: {row['rating_count']}")
#     else:
#         st.warning("Please enter a valid User ID.")

import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("item_based_cf_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

@st.cache_data
def load_data():
    # Apne original dataset ka path yahan do
    column_names = ['user_id', 'product_id', 'rating', 'timestamp']
    df = pd.read_csv("ratings_Electronics.csv", names=column_names, nrows=200000)
    return df

@st.cache_resource
def load_model():
    with open("item_based_cf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
algo_item = load_model()

# ---------------------------
# Helper Functions
# ---------------------------

# Popular products for cold start
def get_popular_products(df, top_n=5):
    product_stats = df.groupby('product_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    product_stats.columns = ['product_id', 'mean_rating', 'rating_count']
    popular_products = product_stats.sort_values(
        by=['rating_count', 'mean_rating'],
        ascending=False
    ).head(top_n)
    return popular_products[['product_id', 'mean_rating', 'rating_count']]

# Similar products using trained item-based CF model
def get_similar_products(product_id, top_n=5):
    try:
        inner_id = algo_item.trainset.to_inner_iid(product_id)
    except ValueError:
        # Cold start for new product
        return get_popular_products(df, top_n)
    
    neighbors = algo_item.get_neighbors(inner_id, k=top_n)
    neighbor_products = [algo_item.trainset.to_raw_iid(inner_id) for inner_id in neighbors]
    return pd.DataFrame({'similar_product_id': neighbor_products})

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🛒 E-Commerce Recommendation System")
st.write("Get personalized or cold start recommendations!")

# Select Recommendation Type
option = st.radio(
    "Choose Recommendation Type:",
    ("User-based (Cold Start handled)", "Product-based (Cold Start handled)")
)

if option == "User-based (Cold Start handled)":
    user_id = st.text_input("Enter User ID:")
    if st.button("Get Recommendations"):
        if not user_id:
            st.warning("Please enter a User ID.")
        elif user_id not in df['user_id'].unique():
            st.info(f"User `{user_id}` not found. Showing popular products.")
            st.dataframe(get_popular_products(df, top_n=5))
        else:
            st.success(f"User `{user_id}` found. Generating personalized recommendations...")
            # NOTE: Yahan tum apna collaborative filtering ka personalized user logic add kar sakte ho
            st.write("⚠ Personalized CF code yahan integrate karna hoga (Surprise prediction).")
            st.dataframe(get_popular_products(df, top_n=5))  # Temporary placeholder

elif option == "Product-based (Cold Start handled)":
    product_id = st.text_input("Enter Product ID:")
    if st.button("Find Similar Products"):
        if not product_id:
            st.warning("Please enter a Product ID.")
        else:
            st.dataframe(get_similar_products(product_id, top_n=5))
