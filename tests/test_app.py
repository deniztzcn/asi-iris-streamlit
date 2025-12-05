from app.predict import predict

def test_predict():
    # Test with a sample Iris Setosa input
    # sepal_length, sepal_width, petal_length, petal_width
    sample_input = [5.1, 3.5, 1.4, 0.2]
    
    # Call prediction function
    species, probabilities = predict(sample_input)
    
    # Assert valid outputs
    assert isinstance(species, str)
    assert species in ["Setosa", "Versicolor", "Virginica"]
    assert len(probabilities) == 3
    assert abs(sum(probabilities) - 1.0) < 0.01
