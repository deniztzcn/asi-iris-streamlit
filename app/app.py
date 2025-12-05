import streamlit as st
from predict import predict


def main():
    st.title("Iris Species Predictor")
    st.write("Enter the flower measurements to predict the Iris species.")

    # Create a form for better user experience
    with st.form("prediction_form"):
        st.subheader("Flower Measurements (in cm)")

        # Create two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.number_input(
                "Sepal Length",
                min_value=0.0,
                max_value=10.0,
                value=5.1,
                step=0.1,
                help="Typical range: 4.3 - 7.9 cm"
            )

            petal_length = st.number_input(
                "Petal Length",
                min_value=0.0,
                max_value=10.0,
                value=1.4,
                step=0.1,
                help="Typical range: 1.0 - 6.9 cm"
            )

        with col2:
            sepal_width = st.number_input(
                "Sepal Width",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
                help="Typical range: 2.0 - 4.4 cm"
            )

            petal_width = st.number_input(
                "Petal Width",
                min_value=0.0,
                max_value=10.0,
                value=0.2,
                step=0.1,
                help="Typical range: 0.1 - 2.5 cm"
            )

        # Submit button
        submitted = st.form_submit_button("Predict Species")

        if submitted:
            # Prepare features for prediction
            features = [sepal_length, sepal_width, petal_length, petal_width]

            # Make prediction
            try:
                species, probabilities = predict(features)

                # Display results
                st.success(f"Predicted Species: **{species}**")

                # Show confidence scores
                st.subheader("Prediction Confidence")
                species_names = ["Setosa", "Versicolor", "Virginica"]

                for name, prob in zip(species_names, probabilities):
                    st.write(f"{name}: {prob:.2%}")
                    st.progress(float(prob))

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


if __name__ == "__main__":
    main()
