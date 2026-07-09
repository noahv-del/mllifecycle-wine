"""
Test script for Wine Quality Prediction API
"""

import requests
import json


def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get("http://localhost:8000/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_single_prediction():
    """Test single wine prediction"""
    print("\n=== Testing Single Prediction ===")

    # Example wine (red wine with moderate quality features)
    wine_data = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "color": "red"
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=wine_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_high_quality_wine():
    """Test prediction for a high quality wine"""
    print("\n=== Testing High Quality Wine ===")

    # High quality wine characteristics (higher alcohol, lower volatile acidity)
    wine_data = {
        "fixed_acidity": 8.5,
        "volatile_acidity": 0.28,
        "citric_acid": 0.56,
        "residual_sugar": 1.8,
        "chlorides": 0.092,
        "free_sulfur_dioxide": 35.0,
        "total_sulfur_dioxide": 103.0,
        "density": 0.9969,
        "pH": 3.3,
        "sulphates": 0.75,
        "alcohol": 12.5,
        "color": "white"
    }

    response = requests.post(
        "http://localhost:8000/predict",
        json=wine_data
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction"""
    print("\n=== Testing Batch Prediction ===")

    wines = [
        {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
            "color": "red"
        },
        {
            "fixed_acidity": 8.5,
            "volatile_acidity": 0.28,
            "citric_acid": 0.56,
            "residual_sugar": 1.8,
            "chlorides": 0.092,
            "free_sulfur_dioxide": 35.0,
            "total_sulfur_dioxide": 103.0,
            "density": 0.9969,
            "pH": 3.3,
            "sulphates": 0.75,
            "alcohol": 12.5,
            "color": "white"
        }
    ]

    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=wines
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def main():
    """Run all tests"""
    print("=" * 60)
    print("Wine Quality Prediction API - Test Suite")
    print("=" * 60)

    try:
        tests = [
            ("Health Check", test_health_check),
            ("Single Prediction", test_single_prediction),
            ("High Quality Wine", test_high_quality_wine),
            ("Batch Prediction", test_batch_prediction)
        ]

        results = []
        for test_name, test_func in tests:
            try:
                passed = test_func()
                results.append((test_name, passed))
            except Exception as e:
                print(f"ERROR in {test_name}: {e}")
                results.append((test_name, False))

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")

        total = len(results)
        passed = sum(1 for _, p in results if p)
        print(f"\nTotal: {passed}/{total} tests passed")

    except requests.exceptions.ConnectionError:
        print("\n ERROR: Could not connect to API.")
        print("Make sure the API is running on http://localhost:8000")
        print("Start it with: python app.py")


if __name__ == "__main__":
    main()
