import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title("Blueberry Yield Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">BlueBerry Yield Predictor</h2>
    </div>  
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    clone_size = st.text_input("The size of blueberry clones", "Type Here")
    honeybee = st.text_input("Density of honey bee species in given area", "Type Here")
    bumbles = st.text_input("Density of bumbles bee species in given area", "Type Here")
    andrena = st.text_input("Density of andrena bee species in given area", "Type Here")
    osmia = st.text_input("Density of osmia bee species in given area", "Type Here")
    Max_Upper_TRange = st.text_input("Maximum Upper Temperature Range", "Type Here")
    Min_Upper_TRange = st.text_input("Minimum Upper Temperature Range", "Type Here")
    Average_Upper_TRange = st.text_input("Average Upper Temperature Range", "Type Here")
    Max_Lower_TRange = st.text_input("Maximum Lower Temperature Range", "Type Here")
    Min_Lower_TRange = st.text_input("Minimum Lower Temperature Range", "Type Here")
    Average_Lower_TRange = st.text_input("Average Lower Temperature Range", "Type Here")
    Raining_Days = st.text_input("Number of days it rains in the given area", "Type Here")
    Average_Raining_Days = st.text_input("Average rainfall in the given area", "Type Here")
    fruit_set = st.text_input("Proportion of flowers that turn into fruit", "Type Here")
    fruit_mass = st.text_input("Mass of the blueberry fruits in the given area", "Type Here")
    seeds = st.text_input("Number of seeds per fruit in the given area", "Type Here")

    results = ""
    if st.button("Predict"):
        data = CustomData(
            clone_size = float(clone_size),
            honeybee = float(honeybee),
            bumbles = float(bumbles),
            andrena = float(andrena),
            osmia = float(osmia),
            Max_Upper_TRange = float(Max_Upper_TRange),
            Min_Upper_TRange = float(Min_Upper_TRange),
            Average_Upper_TRange = float(Average_Upper_TRange),
            Max_Lower_TRange = float(Max_Lower_TRange),
            Min_Lower_TRange = float(Min_Lower_TRange),
            Average_Lower_TRange = float(Average_Lower_TRange),
            Raining_Days = float(Raining_Days),
            Average_Raining_Days = float(Average_Raining_Days),
            fruit_set = float(fruit_set),
            fruit_mass = float(fruit_mass),
            seeds = float(seeds)
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        st.success('The output is {}'.format(results))  

if __name__ == "__main__":
    main()