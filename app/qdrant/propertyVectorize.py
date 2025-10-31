import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

class PropertyVectorizer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_fields = [
            'propertyType', 'source', 'assetType', 'communityType',
            'facing', 'apartmentType', 'stage', 'zone', 'listingType',
            'furnishing', 'status', 'ageOfTheBuilding', 'referredFloorNumber'
        ]
        self.numerical_fields = [
            'rent', 'deposit', 'maintenanceAmount', 'noOfBalconies',
            'noOfBedrooms', 'noOfBathrooms', 'sbua', 'floorNumber',
            'totalFloors', 'carpetArea', 'plotArea', 'lat', 'lng'
        ]
        
    def extract_features(self, property_data):
        """Extract features from property JSON object"""
        features = {}
        
        # Extract numerical features
        features['rent'] = property_data.get('rentalInfo', {}).get('rent', 0)
        features['deposit'] = property_data.get('rentalInfo', {}).get('deposit', 0)
        features['maintenanceAmount'] = property_data.get('rentalInfo', {}).get('maintenanceAmount', 0) or 0
        features['noOfBalconies'] = property_data.get('noOfBalconies', 0) or 0
        features['noOfBedrooms'] = property_data.get('noOfBedrooms', 0) or 0
        features['noOfBathrooms'] = property_data.get('noOfBathrooms', 0) or 0
        features['sbua'] = property_data.get('sbua', 0) or 0
        features['floorNumber'] = property_data.get('floorNumber', 0) or 0
        features['totalFloors'] = property_data.get('totalFloors', 0) or 0
        features['carpetArea'] = property_data.get('carpetArea', 0) or 0
        features['plotArea'] = property_data.get('plotArea', 0) or 0
        
        # Extract geolocation
        geoloc = property_data.get('_geoloc', {})
        features['lat'] = geoloc.get('lat', 0) or 0
        features['lng'] = geoloc.get('lng', 0) or 0
        
        # Extract categorical features
        features['propertyType'] = property_data.get('propertyType', 'unknown')
        features['source'] = property_data.get('source', 'unknown')
        features['assetType'] = property_data.get('assetType', 'unknown')
        features['communityType'] = property_data.get('communityType', 'unknown')
        features['facing'] = property_data.get('facing', 'unknown')
        features['apartmentType'] = property_data.get('apartmentType', 'unknown')
        features['stage'] = property_data.get('stage', 'unknown')
        features['zone'] = property_data.get('zone', 'unknown')
        features['listingType'] = property_data.get('listingType', 'unknown')
        features['furnishing'] = property_data.get('furnishing', 'unknown')
        features['status'] = property_data.get('status', 'unknown')
        features['ageOfTheBuilding'] = property_data.get('ageOfTheBuilding', 'unknown')
        features['referredFloorNumber'] = property_data.get('referredFloorNumber', 'unknown')
        
        # Boolean features
        features['readyToMove'] = 1 if property_data.get('readyToMove') else 0
        
        return features
    
    def fit_transform(self, property_list):
        """
        Fit encoders and transform a list of property objects to vectors
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        # Extract features from all properties
        all_features = [self.extract_features(prop) for prop in property_list]
        
        # Fit label encoders for categorical features
        for field in self.categorical_fields:
            values = [f[field] for f in all_features]
            le = LabelEncoder()
            le.fit(values)
            self.label_encoders[field] = le
        
        # Transform to vectors
        vectors = []
        for features in all_features:
            vector = []
            
            # Add numerical features
            for field in self.numerical_fields:
                vector.append(features[field])
            
            # Add encoded categorical features
            for field in self.categorical_fields:
                encoded = self.label_encoders[field].transform([features[field]])[0]
                vector.append(encoded)
            
            # Add boolean feature
            vector.append(features['readyToMove'])
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.fit_transform(vectors)
        
        return vectors
    
    def transform(self, property_list):
        """
        Transform property objects to vectors using fitted encoders
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, n_features)
        """
        all_features = [self.extract_features(prop) for prop in property_list]
        
        vectors = []
        for features in all_features:
            vector = []
            
            # Add numerical features
            for field in self.numerical_fields:
                vector.append(features[field])
            
            # Add encoded categorical features
            for field in self.categorical_fields:
                # Handle unseen categories
                try:
                    encoded = self.label_encoders[field].transform([features[field]])[0]
                except ValueError:
                    encoded = -1  # Unknown category
                vector.append(encoded)
            
            # Add boolean feature
            vector.append(features['readyToMove'])
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.transform(vectors)
        
        return vectors


# Example usage
if __name__ == "__main__":
    # Sample property data
    property_data = {
        "id": "RNA2424",
        "propertyType": "residential",
        "source": "CRM",
        "rentalInfo": {
            "rent": 45000,
            "deposit": 200000,
            "maintenanceAmount": None
        },
        "assetType": "apartment",
        "noOfBalconies": 2,
        "noOfBedrooms": 3,
        "noOfBathrooms": 3,
        "communityType": "gated",
        "facing": "north",
        "sbua": 1971,
        "floorNumber": 16,
        "apartmentType": "simplex",
        "_geoloc": {
            "lat": 13.0147942,
            "lng": 77.76050099999999
        },
        "stage": "live",
        "zone": "East",
        "listingType": "rental",
        "furnishing": "semi-furnished",
        "readyToMove": True,
        "status": "available",
        "ageOfTheBuilding": "1-5 years",
        "referredFloorNumber": "Higher Floor (10+)"
    }
    
    # Create vectorizer and convert
    vectorizer = PropertyVectorizer()
    
    # For multiple properties
    properties = [property_data]  # Add more property objects here
    
    # Fit and transform
    vectors = vectorizer.fit_transform(properties)
    
    print("Vector shape:", vectors.shape)
    print("Vector:", vectors[0])
    
    # For transforming new data after fitting
    # new_vectors = vectorizer.transform(new_properties)