<?php
// scan.php - Facial Recognition Login Interface
require_once 'config.php';
require_once 'encrypt.php';

// Check if user is already logged in
SessionManager::start();
if (isset($_SESSION['authenticatedUserId'])) {
    header("Location: dashboard.php");
    exit();
}

$error = "";
$success = "";

// Handle facial recognition result from JavaScript
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'verify_face') {
        // CSRF validation
        if (!validateCSRFToken($_POST['csrf_token'] ?? '')) {
            $error = "Invalid request. Please try again.";
        } else {
            // Get data from JavaScript
            $user_id = $_POST['user_id'] ?? '';
            $similarity = floatval($_POST['similarity'] ?? 0);
            $liveness_score = floatval($_POST['liveness_score'] ?? 0);
            $liveness_verified = ($_POST['liveness_verified'] ?? 'false') === 'true';
            
            if (empty($user_id)) {
                $error = "No user ID received from facial recognition.";
            } elseif (!$liveness_verified) {
                $error = "Liveness verification failed. Please ensure you're moving your face naturally.";
                SecurityUtils::logSecurityEvent($user_id, 'liveness_check_failed', 'failure', [
                    'liveness_score' => $liveness_score,
                    'similarity' => $similarity
                ]);
            } elseif ($similarity < Config::FACE_CONFIDENCE_THRESHOLD) {
                $error = "Face recognition confidence too low. Please try again.";
                SecurityUtils::logSecurityEvent($user_id, 'low_confidence_login', 'failure', [
                    'similarity' => $similarity,
                    'threshold' => Config::FACE_CONFIDENCE_THRESHOLD
                ]);
            } else {
                try {
                    $conn = dbConnect(); // Using mysqli connection
                    
                    // Get user details and verify account status
                    $stmt = $conn->prepare("
                        SELECT id, user_id, username, account_status, is_verified, last_login
                        FROM users 
                        WHERE user_id = ?
                    ");
                    $stmt->bind_param("s", $user_id);
                    $stmt->execute();
                    $result = $stmt->get_result();
                    $user = $result->fetch_assoc();
                    $stmt->close();
                    
                    if ($user !== false && $user['account_status'] === 'active' && $user['is_verified']) {
                        // Update last login
                        $stmt = $conn->prepare("UPDATE users SET last_login = NOW() WHERE id = ?");
                        $stmt->bind_param("i", $user['id']);
                        $stmt->execute();
                        $stmt->close();
                        
                        // Set session with basic auth level
                        SessionManager::setAuthLevel('basic', $user['user_id'] ?? $user['username']);
                        $_SESSION['username'] = $user['username'];
                        $_SESSION['user_db_id'] = $user['id'];
                        $_SESSION['authenticatedUserId'] = $user['user_id'] ?? $user['username'];
                        
                        SecurityUtils::logSecurityEvent($user['user_id'], 'facial_login_success', 'success', [
                            'similarity' => $similarity,
                            'liveness_score' => $liveness_score,
                            'login_method' => 'facial_recognition'
                        ]);
                        
                        $success = "Login successful! Redirecting to dashboard...";
                        
                        // JavaScript will handle the redirect
                    } else {
                        $error = "Account not found, not active, or not verified.";
                        SecurityUtils::logSecurityEvent($user_id, 'facial_login_inactive_account', 'failure');
                    }
                    
                    // Close connection
                    $conn->close();
                    
                } catch (Exception $e) {
                    error_log("Facial login error: " . $e->getMessage());
                    $error = "Login failed due to system error.";
                }
            }
        }
    }
}

$csrf_token = generateCSRFToken();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Recognition Login - Secure FacePay</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #3498db;
            --primary-light: #e3f2fd;
            --primary-dark: #2980b9;
            --success: #38a169;
            --warning: #ed8936;
            --error: #e53e3e;
            --text-dark: #2d3748;
            --text-medium: #4a5568;
            --text-light: #718096;
            --border: #e2e8f0;
            --bg-light: #f7fafc;
            --shadow: rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: var(--text-dark);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            width: 100%;
            max-width: 900px;
        }
        
        .scan-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            animation: fadeUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .scan-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), #667eea);
        }
        
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .logo-section {
            margin-bottom: 35px;
        }
        
        .logo {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--primary), #667eea);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .logo i {
            font-size: 32px;
            color: white;
        }
        
        h2 {
            color: var(--text-dark);
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: var(--text-light);
            font-size: 16px;
            margin-bottom: 30px;
        }
        
        /* Steps Section */
        .steps-section {
            margin: 30px 0;
        }
        
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .step-card {
            background: var(--bg-light);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .step-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-color: var(--primary);
        }
        
        .step-number {
            width: 35px;
            height: 35px;
            background: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 16px;
            margin: 0 auto 12px;
        }
        
        .step-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-dark);
        }
        
        .step-description {
            font-size: 12px;
            color: var(--text-medium);
            line-height: 1.4;
        }
        
        /* Camera Section */
        .camera-section {
            margin-bottom: 30px;
        }
        
        .camera-container {
            position: relative;
            width: 100%;
            max-width: 480px;
            margin: 0 auto;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        #video {
            width: 100%;
            height: auto;
            display: block;
            transform: scaleX(-1); /* Mirror effect */
        }
        
        .camera-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }
        
        .face-outline {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 240px;
            border: 3px solid var(--primary);
            border-radius: 50%;
            opacity: 0.8;
            animation: breathe 2s infinite;
        }
        
        @keyframes breathe {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.05); }
        }
        
        .face-outline.success {
            border: 3px solid #27ae60;
            box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.3);
        }
        
        .recording-indicator {
            position: absolute;
            top: 16px;
            left: 16px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #e74c3c;
            display: none;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        
        .recording-indicator.active {
            display: block;
            animation: blink 1s infinite;
        }
        
        /* Status Messages */
        .status-section {
            margin: 25px 0;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .status-message {
            padding: 15px 20px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 12px;
            max-width: 100%;
            text-align: center;
        }
        
        .status-info {
            background: var(--primary-light);
            color: var(--primary);
            border: 1px solid rgba(52, 152, 219, 0.3);
        }
        
        .status-success {
            background: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .status-warning {
            background: #fffbeb;
            color: var(--warning);
            border: 1px solid #fbd38d;
        }
        
        .status-error {
            background: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        /* Progress Bar */
        .progress-container {
            margin: 20px 0;
            width: 100%;
            display: none;
        }
        
        .progress-bar {
            height: 12px;
            background-color: #ecf0f1;
            border-radius: 6px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 25px;
        }
        
        .btn {
            padding: 15px 25px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            min-width: 160px;
            justify-content: center;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-primary:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: var(--bg-light);
            border-color: var(--primary);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background: #2f855a;
            transform: translateY(-2px);
        }
        
        .verify-btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
            gap: 10px;
        }
        
        .verify-btn:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
        }
        
        .verify-btn i {
            font-size: 16px;
        }
        
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Messages from PHP */
        .message {
            padding: 15px 18px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 500;
        }
        
        .success-message {
            background-color: #f0fff4;
            color: var(--success);
            border: 1px solid #c6f6d5;
        }
        
        .error-message {
            background-color: #fff5f5;
            color: var(--error);
            border: 1px solid #fed7d7;
        }
        
        /* Alternative Login */
        .alternative-login {
            margin-top: 30px;
            padding-top: 25px;
            border-top: 1px solid var(--border);
            text-align: center;
        }
        
        .alternative-login p {
            color: var(--text-light);
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        .alt-login-btn {
            background: white;
            color: var(--text-medium);
            border: 2px solid var(--border);
            padding: 12px 20px;
            border-radius: 10px;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .alt-login-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            text-decoration: none;
        }
        
        /* Spinner */
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .scan-container {
                padding: 25px 20px;
            }
            
            .camera-container {
                max-width: 100%;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .btn {
                width: 100%;
                max-width: 250px;
            }
            
            .steps-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="scan-container">
            <div class="logo-section">
                <div class="logo">
                    <i class="fas fa-camera"></i>
                </div>
                <h2>Facial Recognition Login</h2>
                <p class="subtitle">Look at the camera and follow the instructions for secure authentication</p>
            </div>
            
            <?php if ($error): ?>
                <div class="message error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    <?= htmlspecialchars($error) ?>
                </div>
            <?php endif; ?>
            
            <?php if ($success): ?>
                <div class="message success-message">
                    <i class="fas fa-check-circle"></i>
                    <?= htmlspecialchars($success) ?>
                </div>
            <?php endif; ?>
            
            <!-- Process Steps -->
            <div class="steps-section">
                <div class="steps-grid">
                    <div class="step-card">
                        <div class="step-number">1</div>
                        <div class="step-title">Face Detection</div>
                        <div class="step-description">Position your face in the camera frame</div>
                    </div>
                    <div class="step-card">
                        <div class="step-number">2</div>
                        <div class="step-title">Liveness Verification</div>
                        <div class="step-description">Move your face naturally to verify you're real</div>
                    </div>
                    <div class="step-card">
                        <div class="step-number">3</div>
                        <div class="step-title">Authentication</div>
                        <div class="step-description">System recognizes and logs you in</div>
                    </div>
                </div>
            </div>
            
            <!-- Camera Section - Always visible -->
            <div class="camera-section" id="cameraSection">
                <div class="camera-container">
                    <video id="video" autoplay playsinline></video>
                    <div class="camera-overlay">
                        <div class="face-outline" id="faceOutline"></div>
                        <div class="recording-indicator" id="recordingIndicator"></div>
                    </div>
                </div>
            </div>
            
            <!-- Status Messages -->
            <div class="status-section">
                <div class="status-message status-info" id="statusMessage">
                    Ready to start facial recognition login
                </div>
            </div>
            
            <!-- Progress Bar -->
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <button class="btn btn-primary" id="startScanBtn">
                    <i class="fas fa-camera"></i>
                    Start Recognition
                </button>
                
                <!-- Verify Button - Always present but initially hidden -->
                <button class="verify-btn" id="verifyButton" style="display: none;">
                    <i class="fas fa-user-check"></i> Verify & Login
                </button>
                
                <button class="btn btn-secondary" id="retryBtn" style="display: none;">
                    <i class="fas fa-redo"></i>
                    Try Again
                </button>
            </div>
            
            <!-- Hidden form for submission -->
            <form method="POST" id="verificationForm" style="display: none;">
                <input type="hidden" name="csrf_token" value="<?= $csrf_token ?>">
                <input type="hidden" name="action" value="verify_face">
                <input type="hidden" name="user_id" id="hiddenUserId">
                <input type="hidden" name="similarity" id="hiddenSimilarity">
                <input type="hidden" name="liveness_score" id="hiddenLivenessScore">
                <input type="hidden" name="liveness_verified" id="hiddenLivenessVerified">
            </form>
            
            <!-- Alternative Login -->
            <div class="alternative-login">
                <p>Having trouble with facial recognition?</p>
                <a href="login.php" class="alt-login-btn">
                    <i class="fas fa-sign-in-alt"></i>
                    Use Username & Password
                </a>
            </div>
        </div>
    </div>

    <script>
        // Configuration - Same as setup_face.php
        const API_BASE_URL = "http://localhost:5000";
        
        // Global variables - Same approach as setup_face.php
        let stream;
        let frames = [];
        let faceDetected = false;
        let faceDetectionInterval;
        let retryCount = 0;
        const maxRetries = 3;
        
        // Initialize page and camera immediately
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('startScanBtn').addEventListener('click', startFacialScan);
            document.getElementById('retryBtn').addEventListener('click', retryProcess);
            
            // Add event listener for verify button
            document.getElementById('verifyButton').addEventListener('click', function() {
                if (faceDetected) {
                    startLivenessChallenge();
                } else {
                    updateStatus("⚠️ No face detected. Please position your face in the outline.", "warning");
                }
            });
            
            // Initialize camera immediately when page loads
            initializeCameraOnLoad();
        });
        
        // Initialize camera when page loads (without starting scan)
        async function initializeCameraOnLoad() {
            try {
                await initializeWebcam();
                updateStatus('Camera ready! Click "Start Recognition" to begin.', 'info');
            } catch (error) {
                console.error('Error initializing camera on load:', error);
                updateStatus('Camera not available. Click "Start Recognition" to try again.', 'warning');
            }
        }
        
        // Start facial scan - Now just starts detection since camera is already on
        async function startFacialScan() {
            try {
                updateStatus('Starting facial recognition...', 'info');
                
                // If camera isn't already initialized, initialize it
                if (!stream) {
                    await initializeWebcam();
                }
                
                // Start face detection
                startFaceDetection();
                
                // Update UI - hide start button
                document.getElementById('startScanBtn').style.display = 'none';
                
                updateStatus('Position your face in the outline', 'info');
                
            } catch (error) {
                console.error('Error starting facial scan:', error);
                updateStatus('Failed to access camera. Please check permissions.', 'error');
                showRetryOption();
            }
        }
        
        // Initialize webcam - Same as setup_face.php
        async function initializeWebcam() {
            const video = document.getElementById('video');
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 }, 
                    facingMode: 'user' 
                }
            });
            
            video.srcObject = stream;
            
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });
        }
        
        // Start face detection - Same as setup_face.php
        function startFaceDetection() {
            // Check API availability first
            fetch(`${API_BASE_URL}/health`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'healthy' || data.status === 'degraded') {
                        faceDetectionInterval = setInterval(checkFaceDetection, 1000);
                        updateStatus("Position your face in the outline.", "info");
                    } else {
                        updateStatus("⚠️ Face detection API is not fully operational.", "error");
                        showRetryOption();
                    }
                })
                .catch(error => {
                    console.error("API health check failed:", error);
                    updateStatus("❌ Cannot connect to face detection API.", "error");
                    showRetryOption();
                });
        }
        
        // Check face detection - Same as setup_face.php
        function checkFaceDetection() {
            if (!document.getElementById('video').videoWidth) return;
            
            const canvas = document.createElement("canvas");
            canvas.width = document.getElementById('video').videoWidth;
            canvas.height = document.getElementById('video').videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.scale(-1, 1); // Mirror like setup_face.php
            ctx.drawImage(document.getElementById('video'), -canvas.width, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL("image/jpeg").split(',')[1];
            
            fetch(`${API_BASE_URL}/recognize`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: imageData })
            })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                if (data.status === "failed" && (
                    data.message.includes("No face detected") ||
                    data.message.includes("Face extraction failed") ||
                    data.message.includes("Detected face too small") ||
                    data.message.includes("Low face detection confidence"))
                ) {
                    if (faceDetected) {
                        faceDetected = false;
                        document.getElementById('faceOutline').classList.remove('success');
                        updateStatus('Position your face in the outline', 'info');
                        
                        // Hide verify button
                        hideVerifyButton();
                    }
                } else {
                    if (!faceDetected) {
                        faceDetected = true;
                        document.getElementById('faceOutline').classList.add('success');
                        updateStatus('Face detected! Click "Verify & Login" when ready.', 'success');
                        
                        // Show verify button
                        showVerifyButton();
                    }
                }
            })
            .catch(error => {
                console.error("Face detection error:", error);
                faceDetected = false;
                document.getElementById('faceOutline').classList.remove('success');
                
                hideVerifyButton();
            });
        }
        
        // Show verify button
        function showVerifyButton() {
            const verifyButton = document.getElementById('verifyButton');
            verifyButton.style.display = 'inline-flex';
            verifyButton.disabled = !faceDetected;
        }
        
        // Hide verify button
        function hideVerifyButton() {
            const verifyButton = document.getElementById('verifyButton');
            verifyButton.style.display = 'none';
        }
        
        // Start liveness challenge - Same as setup_face.php
        async function startLivenessChallenge() {
            try {
                clearInterval(faceDetectionInterval);
                
                // Hide verify button
                hideVerifyButton();
                
                updateStatus('Ready for liveness challenge! Please move your face naturally.', 'info');
                
                // Start capturing frames automatically after a short delay
                setTimeout(() => {
                    captureFrames();
                }, 2000);
                
            } catch (error) {
                console.error('Error starting liveness challenge:', error);
                updateStatus('Failed to start liveness challenge. Please try again.', 'error');
                showRetryOption();
            }
        }
        
        // Capture frames - Same method as setup_face.php
        function captureFrames() {
            frames = [];
            let captured = 0;
            const total = 20;
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('progressFill').style.width = '0%';
            updateStatus('📸 Capturing frames... Please move your face naturally left and right.', 'info');
            document.getElementById('recordingIndicator').classList.add('active');
            
            const randomChallengeFrame = Math.floor(Math.random() * (total - 5)) + 3;
            let challengeShown = false;
            let challengeType = Math.random() > 0.5 ? 'blink' : 'look_up';
            
            const interval = setInterval(() => {
                const timestamp = new Date().getTime();
                const canvas = document.createElement("canvas");
                canvas.width = document.getElementById('video').videoWidth;
                canvas.height = document.getElementById('video').videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.scale(-1, 1); // Mirror like setup_face.php
                ctx.drawImage(document.getElementById('video'), -canvas.width, 0, canvas.width, canvas.height);
                
                const frameData = {
                    image: canvas.toDataURL("image/jpeg"),
                    timestamp: timestamp,
                    challenge: challengeShown ? challengeType : null
                };
                
                frames.push(frameData);
                captured++;
                document.getElementById('progressFill').style.width = `${(captured / total) * 100}%`;
                
                if (captured === randomChallengeFrame && !challengeShown) {
                    challengeShown = true;
                    if (challengeType === 'blink') {
                        updateStatus("Now, please blink quickly!", "warning");
                    } else {
                        updateStatus("Now, please look up briefly!", "warning");
                    }
                }
                
                if (captured >= total) {
                    clearInterval(interval);
                    verifyWithAPI();
                }
            }, 250);
        }
        
        // Verify with API - Same method as setup_face.php but for login authentication
        function verifyWithAPI() {
            updateStatus('🔍 Analyzing your identity and liveness...', 'info');
            document.getElementById('recordingIndicator').classList.remove('active');
            
            if (frames.length < 6) {
                retryCount++;
                updateStatus("❌ Not enough frames captured. Please try again.", "error");
                handleRetryOrFail();
                return;
            }
            
            const mainImageData = frames[0].image.split(',')[1];
            const additionalFramesData = frames.slice(1).map(frame => frame.image.split(',')[1]);
            const timestamps = frames.map(frame => frame.timestamp);
            const timeDiffs = [];
            for (let i = 1; i < timestamps.length; i++) {
                timeDiffs.push(timestamps[i] - timestamps[i-1]);
            }
            const avgTimeDiff = timeDiffs.reduce((sum, diff) => sum + diff, 0) / timeDiffs.length;
            const timingVariance = Math.sqrt(
                timeDiffs.reduce((sum, diff) => sum + Math.pow(diff - avgTimeDiff, 2), 0) / timeDiffs.length
            );
            
            // Use exact same method as setup_face.php but WITHOUT security_only flag for full authentication
            const authData = {
                image_base64: mainImageData,
                additional_frames: additionalFramesData,
                challenge_type: 'head_movement',
                timing_data: {
                    timestamps: timestamps,
                    avg_diff: avgTimeDiff,
                    variance: timingVariance
                },
                client_random: Math.random().toString().substring(2, 10)
            };
            
            const challengeFrames = frames.filter(frame => frame.challenge !== null);
            if (challengeFrames.length > 0) {
                authData.challenge_data = {
                    type: challengeFrames[0].challenge,
                    frame_index: frames.findIndex(frame => frame.challenge !== null)
                };
            }
            
            // Call /authenticate endpoint for full authentication (login)
            fetch(`${API_BASE_URL}/authenticate`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(authData)
            })
            .then(res => {
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                return res.json();
            })
            .then(data => {
                // For login: check both liveness AND face recognition
                if (data.status === "success" && data.liveness_verified && data.user_id) {
                    // Both security and face recognition passed
                    updateStatus("✅ Authentication successful! Logging you in...", "success");
                    
                    // Submit to PHP
                    document.getElementById('hiddenUserId').value = data.user_id;
                    document.getElementById('hiddenSimilarity').value = data.similarity || 0.9;
                    document.getElementById('hiddenLivenessScore').value = data.liveness_score || 0.8;
                    document.getElementById('hiddenLivenessVerified').value = 'true';
                    
                    setTimeout(() => {
                        document.getElementById('verificationForm').submit();
                    }, 1500);
                } else {
                    retryCount++;
                    // Handle specific error messages like setup_face.php
                    if (!data.liveness_verified) {
                        if (data.message && data.message.includes("spoof")) {
                            updateStatus("🚫 Spoofing attempt detected. Please try again with a real person.", "error");
                        } else if (data.message && data.message.includes("static")) {
                            updateStatus("🚫 Video replay detected. Live verification required.", "error");
                        } else if (data.message && data.message.includes("Static image detected")) {
                            updateStatus("❌ Static image detected! Please use a live camera.", "error");
                        } else if (data.message && data.message.includes("Video replay detected")) {
                            updateStatus("❌ Video replay detected! Please use a live camera.", "error");
                        } else if (data.message && data.message.includes("Deepfake detected")) {
                            updateStatus("❌ Deepfake detected! Please use your real face.", "error");
                        } else {
                            updateStatus(`❌ Security check failed: ${data.message || "Please follow the instructions carefully."}`, "error");
                        }
                    } else if (!data.user_id) {
                        updateStatus("❌ Face not recognized. Please ensure you have registered your face first.", "error");
                    } else {
                        updateStatus(`❌ Authentication failed: ${data.message || "Unknown error"}`, "error");
                    }
                    handleRetryOrFail();
                }
            })
            .catch(error => {
                console.error("API error:", error);
                retryCount++;
                updateStatus("❌ Server error during authentication. Please try again later.", "error");
                handleRetryOrFail();
            });
        }
        
        // Handle retry or fail - Same as setup_face.php
        function handleRetryOrFail() {
            if (retryCount >= maxRetries) {
                updateStatus("⚠️ Too many failed attempts. Please try again later.", "error");
                setTimeout(() => {
                    window.location.href = 'login.php';
                }, 3000);
            } else {
                showRetryOption();
            }
        }
        
        // Update status - Same as setup_face.php
        function updateStatus(message, type = 'info') {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message status-${type}`;
        }
        
        // Show retry option - Same as setup_face.php
        function showRetryOption() {
            document.getElementById('retryBtn').style.display = 'inline-flex';
        }
        
        // Retry process - Same as setup_face.php
        function retryProcess() {
            // Reset everything
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
            
            // Reset UI - keep camera visible
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            
            // Reset buttons
            document.getElementById('startScanBtn').style.display = 'inline-flex';
            document.getElementById('retryBtn').style.display = 'none';
            hideVerifyButton();
            
            // Reset variables
            frames = [];
            faceDetected = false;
            retryCount = 0;
            
            updateStatus('Ready to try again. Click "Start Recognition" to begin.', 'info');
        }
        
        // Handle successful PHP response
        <?php if ($success): ?>
        setTimeout(() => {
            window.location.href = 'dashboard.php';
        }, 3000);
        <?php endif; ?>
        
        // Clean up on page unload - Same as setup_face.php
        window.addEventListener('beforeunload', () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
        });
    </script>
</body>
</html>