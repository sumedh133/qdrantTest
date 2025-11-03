import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hashlib

class PropertyVectorizer:
    def __init__(self, target_dim=128):
        self.target_dim = target_dim
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
        
    def _safe_get_number(self, value, default=0):
        """Safely convert value to number, handling None and invalid values"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_get_string(self, value, default='unknown'):
        """Safely convert value to string, handling None"""
        if value is None or value == '':
            return default
        return str(value)
    
    def extract_features(self, property_data):
        """Extract features from property JSON object"""
        if property_data is None:
            property_data = {}
            
        features = {}
        
        # Extract numerical features with None handling
        rental_info = property_data.get('rentalInfo') or {}
        features['rent'] = self._safe_get_number(rental_info.get('rent'), 0)
        features['deposit'] = self._safe_get_number(rental_info.get('deposit'), 0)
        features['maintenanceAmount'] = self._safe_get_number(rental_info.get('maintenanceAmount'), 0)
        features['noOfBalconies'] = self._safe_get_number(property_data.get('noOfBalconies'), 0)
        features['noOfBedrooms'] = self._safe_get_number(property_data.get('noOfBedrooms'), 0)
        features['noOfBathrooms'] = self._safe_get_number(property_data.get('noOfBathrooms'), 0)
        features['sbua'] = self._safe_get_number(property_data.get('sbua'), 0)
        features['floorNumber'] = self._safe_get_number(property_data.get('floorNumber'), 0)
        features['totalFloors'] = self._safe_get_number(property_data.get('totalFloors'), 0)
        features['carpetArea'] = self._safe_get_number(property_data.get('carpetArea'), 0)
        features['plotArea'] = self._safe_get_number(property_data.get('plotArea'), 0)
        
        # Extract geolocation with None handling
        geoloc = property_data.get('_geoloc') or {}
        features['lat'] = self._safe_get_number(geoloc.get('lat'), 0)
        features['lng'] = self._safe_get_number(geoloc.get('lng'), 0)
        
        # Extract categorical features with None handling
        features['propertyType'] = self._safe_get_string(property_data.get('propertyType'))
        features['source'] = self._safe_get_string(property_data.get('source'))
        features['assetType'] = self._safe_get_string(property_data.get('assetType'))
        features['communityType'] = self._safe_get_string(property_data.get('communityType'))
        features['facing'] = self._safe_get_string(property_data.get('facing'))
        features['apartmentType'] = self._safe_get_string(property_data.get('apartmentType'))
        features['stage'] = self._safe_get_string(property_data.get('stage'))
        features['zone'] = self._safe_get_string(property_data.get('zone'))
        features['listingType'] = self._safe_get_string(property_data.get('listingType'))
        features['furnishing'] = self._safe_get_string(property_data.get('furnishing'))
        features['status'] = self._safe_get_string(property_data.get('status'))
        features['ageOfTheBuilding'] = self._safe_get_string(property_data.get('ageOfTheBuilding'))
        features['referredFloorNumber'] = self._safe_get_string(property_data.get('referredFloorNumber'))
        
        # Boolean features with None handling
        ready_to_move = property_data.get('readyToMove')
        features['readyToMove'] = 1 if ready_to_move is True else 0
        
        return features
    
    def _safe_divide(self, numerator, denominator, default=0):
        """Safely divide two numbers, handling division by zero and None"""
        try:
            num = self._safe_get_number(numerator, 0)
            denom = self._safe_get_number(denominator, 0)
            if denom == 0:
                return default
            return num / denom
        except (ValueError, TypeError, ZeroDivisionError):
            return default
    
    def _categorical_to_embeddings(self, value, num_dims=8):
        """Convert categorical value to fixed-size embedding using hashing"""
        # Create a hash of the categorical value
        hash_obj = hashlib.md5(str(value).encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to array of floats between -1 and 1
        embedding = []
        for i in range(num_dims):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Normalize to [-1, 1]
            embedding.append((byte_val / 127.5) - 1.0)
        
        return embedding
    
    def fit_transform(self, property_list):
        """
        Fit encoders and transform a list of property objects to vectors
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, target_dim)
        """
        # Extract features from all properties
        all_features = [self.extract_features(prop) for prop in property_list]
        
        # Transform to vectors
        vectors = []
        for features in all_features:
            vector = []
            
            # Add numerical features (13 features)
            for field in self.numerical_fields:
                vector.append(features[field])
            
            # Add categorical features as embeddings (13 fields × 8 dims = 104 features)
            for field in self.categorical_fields:
                cat_embedding = self._categorical_to_embeddings(features[field], num_dims=8)
                vector.extend(cat_embedding)
            
            # Add boolean feature (1 feature)
            vector.append(features['readyToMove'])
            
            # Total so far: 13 + 104 + 1 = 118 features
            # Pad to 128 with derived features
            if len(vector) < self.target_dim:
                # Add some derived features with safe division
                rent = features['rent']
                sbua = features['sbua']
                bedrooms = features['noOfBedrooms']
                
                # Rent per sqft
                vector.append(self._safe_divide(rent, sbua))
                # Rent per bedroom
                vector.append(self._safe_divide(rent, bedrooms))
                # Deposit to rent ratio
                vector.append(self._safe_divide(features['deposit'], rent))
                # Floor ratio
                total_floors = features['totalFloors']
                vector.append(self._safe_divide(features['floorNumber'], total_floors))
                
                # Bathrooms per bedroom
                bathrooms = features['noOfBathrooms']
                vector.append(self._safe_divide(bathrooms, bedrooms))
                
                # Area ratios
                carpet = features['carpetArea']
                plot = features['plotArea']
                vector.append(self._safe_divide(carpet, sbua))
                vector.append(self._safe_divide(plot, sbua))
                
                # Location hash features (3 more to reach 128)
                lat_lng_str = f"{features['lat']:.4f},{features['lng']:.4f}"
                loc_embedding = self._categorical_to_embeddings(lat_lng_str, num_dims=3)
                vector.extend(loc_embedding)
            
            # Ensure exact 128 dimensions
            if len(vector) < self.target_dim:
                vector.extend([0.0] * (self.target_dim - len(vector)))
            elif len(vector) > self.target_dim:
                vector = vector[:self.target_dim]
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.fit_transform(vectors)
        
        return vectors
    
    def transform(self, property_list):
        """
        Transform property objects to vectors using fitted scaler
        
        Args:
            property_list: List of property JSON objects
            
        Returns:
            numpy array of shape (n_samples, target_dim)
        """
        all_features = [self.extract_features(prop) for prop in property_list]
        
        vectors = []
        for features in all_features:
            vector = []
            
            # Add numerical features
            for field in self.numerical_fields:
                vector.append(features[field])
            
            # Add categorical features as embeddings
            for field in self.categorical_fields:
                cat_embedding = self._categorical_to_embeddings(features[field], num_dims=8)
                vector.extend(cat_embedding)
            
            # Add boolean feature
            vector.append(features['readyToMove'])
            
            # Add derived features (same as fit_transform)
            if len(vector) < self.target_dim:
                rent = features['rent']
                sbua = features['sbua']
                bedrooms = features['noOfBedrooms']
                
                vector.append(self._safe_divide(rent, sbua))
                vector.append(self._safe_divide(rent, bedrooms))
                vector.append(self._safe_divide(features['deposit'], rent))
                total_floors = features['totalFloors']
                vector.append(self._safe_divide(features['floorNumber'], total_floors))
                
                bathrooms = features['noOfBathrooms']
                vector.append(self._safe_divide(bathrooms, bedrooms))
                
                carpet = features['carpetArea']
                plot = features['plotArea']
                vector.append(self._safe_divide(carpet, sbua))
                vector.append(self._safe_divide(plot, sbua))
                
                lat_lng_str = f"{features['lat']:.4f},{features['lng']:.4f}"
                loc_embedding = self._categorical_to_embeddings(lat_lng_str, num_dims=3)
                vector.extend(loc_embedding)
            
            # Ensure exact 128 dimensions
            if len(vector) < self.target_dim:
                vector.extend([0.0] * (self.target_dim - len(vector)))
            elif len(vector) > self.target_dim:
                vector = vector[:self.target_dim]
            
            vectors.append(vector)
        
        # Convert to numpy array and scale
        vectors = np.array(vectors)
        vectors = self.scaler.transform(vectors)
        
        return vectors


# Example usage
if __name__ == "__main__":
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
    
    vectorizer = PropertyVectorizer(target_dim=128)
    properties = [property_data]
    vectors = vectorizer.fit_transform(properties)
    
    print("Vector shape:", vectors.shape)
    print("Vector dimensions:", len(vectors[0]))
    print("First 10 values:", vectors[0][:10])