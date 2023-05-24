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
    author = st.selectbox("Author",
                          ('Thompson', 'Janssen', 'Weatherhead', 'Beus', 'Peskov', 'Williams', 'Richenderfer', 'Mortimore', 'Kossolapov', 'Inasaka'),
                          label_visibility = "visible",
                          disabled = False,
                          )
    geometry = st.selectbox("Geometry",
                          ('tube', 'annulus', 'plate'),
                          label_visibility = "visible",
                          disabled = False,
                          )
    pressure = st.text_input("Pressure", "Type Here")
    mass_flux = st.text_input("Mass Flux", "Type Here")
    d_e = st.text_input("Diameter_e", "Type Here")
    d_h = st.text_input("Diameter_h", "Type Here")
    length = st.text_input("Length", "Type Here")
    chf_exp = st.text_input("Intensity", "Type Here")

    print(author, geometry)
    results = ""
    if st.button("Predict"):
        data = CustomData(
            author = str(author),
            geometry = str(geometry),
            pressure = float(pressure),
            mass_flux = float(mass_flux),
            d_e = float(d_e),
            d_h = float(d_h),
            length = float(length),
            chf_exp = float(chf_exp),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        st.success('The output is {}'.format(results))  

if __name__ == "__main__":
    main()