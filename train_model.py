import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


def main():
    print("Loading Iris dataset...")
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create and train the classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Create app/ folder if it doesn't exist
    os.makedirs("app", exist_ok=True)

    # Save the trained model in the app/ folder
    model_path = "app/model.joblib"
    print(f"\nSaving model to {model_path}...")
    joblib.dump(clf, model_path)

    # Verify the file was created
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✓ Model saved successfully! (File size: {file_size:,} bytes)")
    else:
        print("✗ Error: Model file was not created")

    return accuracy


if __name__ == "__main__":
    main()