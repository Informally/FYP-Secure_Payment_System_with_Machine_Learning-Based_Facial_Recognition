# Secure Payment System with Machine Learning-Based Facial Recognition

A comprehensive secure payment system that integrates machine learning-based facial recognition technology with multi-layered security protocols for self-service payment environments such as kiosks.

## Features

- **Facial Recognition Authentication**: Uses MTCNN for face detection and FaceNet for feature extraction
- **Multi-Method Liveness Detection**: Blink detection, head movement analysis, optical flow analysis, and custom anti-spoofing model
- **Hybrid Classification**: Cosine similarity for small user bases, SVM for larger datasets
- **Secure Data Protection**: AES-256-CBC encryption for financial data, BCrypt hashing for credentials
- **Merchant Integration**: Gateway architecture for easy business integration
- **Admin Panel**: Complete user and merchant management system

## System Requirements

- **XAMPP** (Apache, MySQL, PHP)
- **Python 3.7+**
- **Webcam** for facial recognition

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Informally/FYP-Secure_Payment_System_with_Machine_Learning-Based_Facial_Recognition.git
cd FYP-Secure_Payment_System_with_Machine_Learning-Based_Facial_Recognition
```

### 2. Database Setup
1. Start XAMPP and ensure Apache and MySQL services are running
2. Open phpMyAdmin in your browser: `http://localhost/phpmyadmin`
3. Create a new database named `payment_facial`
4. Import the database file located in `Database_Import/payment_facial.sql`

### 3. Python Dependencies Installation
Navigate to the face recognition directory and install required packages:
```bash
cd face_recognition
pip install flask opencv-python torch torchvision facenet-pytorch mtcnn scikit-learn mysql-connector-python flask-cors dlib numpy pillow
```

**Note**: You'll also need to download the dlib facial landmark predictor:
- Download `shape_predictor_68_face_landmarks.dat` from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract and place it in the `face_recognition` directory

### 4. Start the Facial Recognition API
```bash
cd face_recognition
python face_api5.py
```
The API will run on `http://localhost:5000`

## Usage

### Admin Access
- **URL**: `http://localhost/FYP/admin_side/login.php`
- **Username**: `admin`
- **Password**: `Admin123!`

**Admin Features**:
- Merchant management and registration
- User management and monitoring
- Transaction oversight and reporting
- System analytics

### Customer Access
- **URL**: `http://localhost/FYP/customer_side/login.php`
- Register a new account or login with existing credentials
- Option to login with facial recognition (after face registration)

**Customer Features**:
- Account registration with email verification
- Wallet management and top-up functionality
- Facial recognition setup for payments
- Transaction history and profile management

### Merchant Demo
- **URL**: `http://localhost/FYP/business_integration/e-commerce.php`
- Demonstrates how businesses can integrate the payment system
- Shows the complete customer journey from product selection to payment

## System Architecture

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: PHP (XAMPP) + Python Flask API
- **Database**: MySQL
- **Machine Learning**: MTCNN, FaceNet, SVM, Custom Anti-spoofing Model
- **Security**: AES-256-CBC encryption, BCrypt hashing, Multi-method liveness detection

## Payment Flow

1. **Face Scanning**: Customer scans face at merchant kiosk
2. **Liveness Detection**: System verifies real person using multiple methods
3. **Identity Verification**: Facial recognition confirms user identity
4. **PIN Authentication**: Customer enters 6-digit PIN for two-factor authentication
5. **Transaction Processing**: Payment is processed if sufficient wallet balance
6. **Confirmation**: User redirected back to merchant with transaction status

## Security Features

- **Multi-layered Liveness Detection**: Prevents spoofing attacks using photos/videos
- **Encrypted Financial Data**: User wallet balances protected with AES-256-CBC
- **Secure Credential Storage**: Passwords and PINs hashed with BCrypt
- **Session Management**: Secure session handling with timeout mechanisms
- **Rate Limiting**: Prevents brute force and spam attempts

## Development Notes

- System designed for local development and testing
- Facial recognition API must be running for authentication features to work
- Default welcome bonus of RM 50.00 credited to new accounts for testing
- Merchant integration uses API key authentication for secure transactions

## Troubleshooting

- Ensure XAMPP services (Apache & MySQL) are running
- Verify Python Flask API is active on port 5000
- Check that webcam permissions are granted for facial recognition
- Confirm database connection settings in configuration files

## Contributing

This project was developed as a Final Year Project (FYP) demonstrating the practical implementation of secure facial recognition in payment systems for self-service environments.

## License

This project is for educational and research purposes.
