import numpy as np
from sklearn.preprocessing import StandardScaler

class PropertyVectorizer:
    def __init__(self):
        self.target_dim = 8
        self.scaler = StandardScaler()
        
    def _safe_get_number(self, value, default=0):
        """Safely convert value to number, handling None and invalid values"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def extract_features(self, property_data):
        """Extract the 7 specified features from property JSON object"""
        if property_data is None:
            property_data = {}
            
        features = {}
        
        # Extract geolocation (2 features)
        geoloc = property_data.get('_geoloc') or {}
        features['lat'] = self._safe_get_number(geoloc.get('lat'), 0)
        features['lng'] = self._safe_get_number(geoloc.get('lng'), 0)
        
        # Extract property attributes (3 features)
        features['noOfBedrooms'] = self._safe_get_number(property_data.get('noOfBedrooms'), 0)
        features['sbua'] = self._safe_get_number(property_data.get('sbua'), 0)
        features['carpetArea'] = self._safe_get_number(property_data.get('carpetArea'), 0)
        features['plotSize'] = self._safe_get_number(property_data.get('plotSize'), 0)
        
        # Extract pricing (1 feature)
        pricing = property_data.get('pricing') or {}
        features['totalAskPrice'] = self._safe_get_number(pricing.get('totalAskPrice'), 0)
        
        return features
    
    def _safe_divide(self, numerator, denominator, default=0):
        """Safely divide two numbers, handling division by zero"""
        try:
            num = self._safe_get_number(numerator, 0)
            denom = self._safe_get_number(denominator, 0)
            if denom == 0:
                return default
            return num / denom
        except (ValueError, TypeError, ZeroDivisionError):
            return default
    
    def fit_transform(self, property_list):
        """
        Fit scaler and transform properties to 8D vectors
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, 8)
        """
        all_features = [self.extract_features(prop) for prop in property_list]
        
        vectors = []
        for features in all_features:
            vector = []
            
            # Base features (7 dimensions)
            vector.append(features['lat'])
            vector.append(features['lng'])
            vector.append(features['noOfBedrooms'])
            vector.append(features['sbua'])
            vector.append(features['carpetArea'])
            vector.append(features['plotSize'])
            vector.append(features['totalAskPrice'])
            
            # Derived feature (1 dimension) - price per sqft
            price_per_sqft = self._safe_divide(
                features['totalAskPrice'], 
                features['sbua']
            )
            vector.append(price_per_sqft)
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.fit_transform(vectors)
        
        return vectors
    
    def transform(self, property_list):
        """
        Transform properties to 8D vectors using fitted scaler
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, 8)
        """
        all_features = [self.extract_features(prop) for prop in property_list]
        
        vectors = []
        for features in all_features:
            vector = []
            
            # Base features (7 dimensions)
            vector.append(features['lat'])
            vector.append(features['lng'])
            vector.append(features['noOfBedrooms'])
            vector.append(features['sbua'])
            vector.append(features['carpetArea'])
            vector.append(features['plotSize'])
            vector.append(features['totalAskPrice'])
            
            # Derived feature (1 dimension)
            price_per_sqft = self._safe_divide(
                features['totalAskPrice'], 
                features['sbua']
            )
            vector.append(price_per_sqft)
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.transform(vectors)
        
        return vectors


# Example usage
if __name__ == "__main__":
    property_data = {
        "id": "RNA2424",
        "noOfBedrooms": 3,
        "sbua": 1971,
        "carpetArea": 1800,
        "plotSize": 2500,
        "_geoloc": {
            "lat": 13.0147942,
            "lng": 77.76050099999999
        },
        "pricing": {
            "totalAskPrice": 8500000
        }
    }
    
    vectorizer = PropertyVectorizer()
    properties = [property_data]
    vectors = vectorizer.fit_transform(properties)
    
    print("Vector shape:", vectors.shape)
    print("Vector dimensions:", len(vectors[0]))
    print("8D Vector:", vectors[0])
    print("\nFeature breakdown:")
    print("1. Latitude (scaled)")
    print("2. Longitude (scaled)")
    print("3. Number of Bedrooms (scaled)")
    print("4. SBUA (scaled)")
    print("5. Carpet Area (scaled)")
    print("6. Plot Size (scaled)")
    print("7. Total Ask Price (scaled)")
    print("8. Price per Sqft (scaled)")