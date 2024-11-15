import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

def main():
    st.title("Sentiment Analysis using Natural Language Processing")
    st.subheader("Simple Sequence Classification")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter the text you want to analyze here: ")
            submit_button = st.form_submit_button(label='Analyze')

        st.write("Select a range of values to segregate the classes:")
        st.caption("(The region to the left of the left pointer shows that sentences with polarity in that region may "
                 "have a negative sentiment and the region towards the right of the right pointer shows a positive "
                 "sentiment leaving the one in the middle to be neutral sentiments)")
        values = st.slider('Select a range of polarity values for neutral sentiments', -1.0, 1.0, (-0.05, 0.05))

        # layout
        col1, col2 = st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Sentiment
                if sentiment.polarity > values[1]:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < values[0]:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")
        st.write("This is a Sequence Classification application that deals with sentence level sentiment analysis by classifying sentences based on tokenizing user inputs.")

if __name__ == '__main__':
    main()