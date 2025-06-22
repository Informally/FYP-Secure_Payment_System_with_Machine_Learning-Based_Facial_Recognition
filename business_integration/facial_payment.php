<?php
require_once '../customer_side/config.php';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Payment - Prototype Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: #f8f8f8;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .kiosk-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            padding: 15px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .kiosk-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .payment-container {
            max-width: 1000px;
            margin: 20px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            flex: 1;
        }

        .facial-scan-section {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 3px solid #27ae60;
        }

        .section-header {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
            padding: 20px;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .scan-content {
            padding: 30px;
            text-align: center;
        }

        .scan-steps {
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .scan-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            opacity: 0.5;
            transition: all 0.3s ease;
        }

        .scan-step.active {
            opacity: 1;
            transform: scale(1.1);
        }

        .scan-step.completed {
            opacity: 1;
            color: #27ae60;
        }

        .step-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #ecf0f1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: #7f8c8d;
        }

        .scan-step.active .step-icon {
            background: #3498db;
            color: white;
        }

        .scan-step.completed .step-icon {
            background: #27ae60;
            color: white;
        }

        .step-text {
            font-size: 14px;
            font-weight: bold;
            text-align: center;
        }

        .webcam-container {
            position: relative;
            width: 100%;
            max-width: 500px;
            margin: 0 auto 30px;
            border-radius: 15px;
            overflow: hidden;
            border: 4px solid #27ae60;
        }

        #video {
            width: 100%;
            height: auto;
            display: block;
        }

        .face-guide {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }

        .face-outline {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 260px;
            border: 3px dashed #e74c3c;
            border-radius: 120px 120px 80px 80px;
            transition: all 0.3s ease;
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

        .challenge-instruction {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 2px solid #f39c12;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #d68910;
        }

        .challenge-instruction.active {
            background: linear-gradient(135deg, #d1ecf1, #a8dadc);
            border-color: #3498db;
            color: #2980b9;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        .status-message {
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }

        .status-info {
            background: #e3f2fd;
            color: #1976d2;
            border: 2px solid #2196f3;
        }

        .status-success {
            background: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #4caf50;
        }

        .status-warning {
            background: #fff8e1;
            color: #f57c00;
            border: 2px solid #ff9800;
        }

        .status-error {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #f44336;
        }

        .progress-container {
            margin: 20px 0;
            width: 100%;
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

        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3);
        }

        .btn-success {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
        }

        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(39, 174, 96, 0.3);
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
            transform: translateY(-2px);
        }

        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .verify-btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 30px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
            margin: 20px auto;
        }
        
        .verify-btn:hover {
            background-color: #45a049;
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(76, 175, 80, 0.3);
        }
        
        .verify-btn i {
            margin-right: 10px;
        }
        
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Order Summary Sidebar */
        .order-sidebar {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 3px solid #ffc72c;
            height: fit-content;
        }

        .order-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            color: white;
            padding: 20px;
            font-size: 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .order-content {
            padding: 20px;
        }

        .order-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }

        .order-item:last-child {
            border-bottom: none;
        }

        .item-icon {
            font-size: 24px;
            width: 40px;
            text-align: center;
        }

        .item-info {
            flex: 1;
        }

        .item-name {
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 2px;
        }

        .item-price {
            color: #da291c;
            font-weight: bold;
            font-size: 13px;
        }

        .item-qty {
            background: #da291c;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: bold;
        }

        .order-total {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }

        .total-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .total-row.final {
            font-size: 18px;
            font-weight: bold;
            border-top: 1px solid rgba(255,255,255,0.3);
            padding-top: 10px;
            margin-top: 10px;
        }

        @media (max-width: 768px) {
            .payment-container {
                grid-template-columns: 1fr;
                gap: 20px;
            }

            .kiosk-header h1 {
                font-size: 2rem;
            }

            .scan-steps {
                flex-direction: column;
                gap: 15px;
            }

            .action-buttons {
                flex-direction: column;
            }
        }

        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-user-check"></i> Facial Payment - Prototype Kiosk</h1>
    </div>

    <div class="payment-container">
        <!-- Facial Scan Section -->
        <div class="facial-scan-section">
            <div class="section-header">
                <i class="fas fa-camera"></i>
                Facial Recognition Payment
            </div>
            <div class="scan-content">
                <!-- Progress Steps -->
                <div class="scan-steps">
                    <div class="scan-step active" id="step1">
                        <div class="step-icon">
                            <i class="fas fa-camera"></i>
                        </div>
                        <div class="step-text">Position Face</div>
                    </div>
                    <div class="scan-step" id="step2">
                        <div class="step-icon">
                            <i class="fas fa-user-check"></i>
                        </div>
                        <div class="step-text">Liveness Check</div>
                    </div>
                    <div class="scan-step" id="step3">
                        <div class="step-icon">
                            <i class="fas fa-lock"></i>
                        </div>
                        <div class="step-text">Enter PIN</div>
                    </div>
                </div>

                <!-- Challenge Instruction -->
                <div class="challenge-instruction" id="challengeInstruction" style="display: none;">
                    Please slowly turn your head left and right to verify your identity
                </div>

                <!-- Webcam Container -->
                <div class="webcam-container" id="webcamContainer" style="display: none;">
                    <video id="video" autoplay playsinline></video>
                    <div class="face-guide">
                        <div class="face-outline" id="faceOutline"></div>
                        <div class="recording-indicator" id="recordingIndicator"></div>
                    </div>
                </div>

                <!-- Status Message -->
                <div class="status-message status-info" id="statusMessage">
                    Welcome! Click "Start Scan" to begin facial recognition payment
                </div>

                <!-- Progress Bar -->
                <div class="progress-container" id="progressContainer" style="display: none;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="action-buttons">
                    <button class="btn btn-primary" id="startScanBtn" onclick="startFacialScan()">
                        <i class="fas fa-play"></i> Start Scan
                    </button>
                    <button class="btn btn-primary" id="retryBtn" style="display: none;" onclick="retryProcess()">
                        <i class="fas fa-redo"></i> Try Again
                    </button>
                    <button class="btn btn-secondary" onclick="goBackToCheckout()">
                        <i class="fas fa-arrow-left"></i> Back to Checkout
                    </button>
                </div>
            </div>
        </div>

        <!-- Order Summary Sidebar -->
        <div class="order-sidebar">
            <div class="order-header">
                <i class="fas fa-receipt"></i>
                Order Summary
            </div>
            <div class="order-content">
                <div id="orderItems">
                    <!-- Order items will be loaded here -->
                </div>

                <div class="order-total">
                    <div class="total-row">
                        <span>Subtotal:</span>
                        <span>RM <span id="subtotal">0.00</span></span>
                    </div>
                    <div class="total-row">
                        <span>Tax (6% SST):</span>
                        <span>RM <span id="tax">0.00</span></span>
                    </div>
                    <div class="total-row final">
                        <span>Total:</span>
                        <span>RM <span id="finalTotal">0.00</span></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let cart = [];
        let stream;
        let frames = [];
        let currentStep = 1;
        let faceDetected = false;
        let faceDetectionInterval;
        let userId = null;
        let retryCount = 0;
        const maxRetries = 3;

        const API_BASE_URL = "http://localhost:5000";

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            loadCartFromStorage();
            renderOrderSummary();
        });

        function loadCartFromStorage() {
            const savedCart = sessionStorage.getItem('kioskCart');
            if (savedCart) {
                cart = JSON.parse(savedCart);
            }

            if (cart.length === 0) {
                updateStatus('Your cart is empty. Redirecting to menu...', 'error');
                setTimeout(() => {
                    window.location.href = 'e-commerce.php';
                }, 2000);
            }
        }

        function renderOrderSummary() {
            const orderItems = document.getElementById('orderItems');
            const subtotalEl = document.getElementById('subtotal');
            const taxEl = document.getElementById('tax');
            const finalTotalEl = document.getElementById('finalTotal');

            // Render order items
            orderItems.innerHTML = cart.map(item => `
                <div class="order-item">
                    <div class="item-icon">${item.icon}</div>
                    <div class="item-info">
                        <div class="item-name">${item.name}</div>
                        <div class="item-price">RM ${item.price.toFixed(2)}</div>
                    </div>
                    <div class="item-qty">${item.quantity}</div>
                </div>
            `).join('');

            // Calculate totals
            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.06;
            const finalTotal = subtotal + tax;

            subtotalEl.textContent = subtotal.toFixed(2);
            taxEl.textContent = tax.toFixed(2);
            finalTotalEl.textContent = finalTotal.toFixed(2);
        }

        async function startFacialScan() {
            try {
                updateStep(1);
                updateStatus('Starting facial scan...', 'info');
                
                // Initialize webcam
                await initializeWebcam();
                
                // Start face detection
                startFaceDetection();
                
                // Update UI
                document.getElementById('startScanBtn').style.display = 'none';
                document.getElementById('webcamContainer').style.display = 'block';
                
                updateStatus('Position your face in the outline', 'info');
                
            } catch (error) {
                console.error('Error starting facial scan:', error);
                updateStatus('Failed to access camera. Please check permissions.', 'error');
                showRetryOption();
            }
        }

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

        function checkFaceDetection() {
            if (!document.getElementById('video').videoWidth) return;
            
            const canvas = document.createElement("canvas");
            canvas.width = document.getElementById('video').videoWidth;
            canvas.height = document.getElementById('video').videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(document.getElementById('video'), 0, 0);
            
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
                        
                        // Remove verify button if it exists
                        const existingVerifyBtn = document.getElementById('verifyButton');
                        if (existingVerifyBtn) {
                            existingVerifyBtn.disabled = true;
                        }
                    }
                } else {
                    if (!faceDetected) {
                        faceDetected = true;
                        document.getElementById('faceOutline').classList.add('success');
                        updateStatus('Face detected! Click "Verify" when ready.', 'success');
                        
                        // Show verify button
                        showVerifyButton();
                    }
                }
            })
            .catch(error => {
                console.error("Face detection error:", error);
                faceDetected = false;
                document.getElementById('faceOutline').classList.remove('success');
                
                const existingVerifyBtn = document.getElementById('verifyButton');
                if (existingVerifyBtn) {
                    existingVerifyBtn.disabled = true;
                }
            });
        }

        function showVerifyButton() {
            // Check if verify button already exists
            let verifyButton = document.getElementById('verifyButton');
            
            if (!verifyButton) {
                const verifyButtonContainer = document.createElement('div');
                verifyButtonContainer.style.textAlign = 'center';
                verifyButtonContainer.style.margin = '20px 0';

                verifyButton = document.createElement('button');
                verifyButton.innerHTML = '<i class="fas fa-camera"></i> Verify';
                verifyButton.className = 'verify-btn';
                verifyButton.id = 'verifyButton';

                verifyButtonContainer.appendChild(verifyButton);
                document.getElementById('statusMessage').parentNode.insertBefore(verifyButtonContainer, document.getElementById('progressContainer'));

                verifyButton.addEventListener('click', function () {
                    if (faceDetected) {
                        startLivenessChallenge();
                    } else {
                        updateStatus("⚠️ No face detected. Please position your face in the outline.", "warning");
                    }
                });
            }
            
            verifyButton.disabled = !faceDetected;
        }

        async function startLivenessChallenge() {
            try {
                clearInterval(faceDetectionInterval);
                updateStep(2);
                
                // Show challenge instruction
                const instructionEl = document.getElementById('challengeInstruction');
                instructionEl.style.display = 'block';
                instructionEl.classList.add('active');
                
                // Hide verify button
                const verifyButton = document.getElementById('verifyButton');
                if (verifyButton) {
                    verifyButton.style.display = 'none';
                }
                
                updateStatus('Ready for liveness challenge! Follow the instruction above.', 'info');
                
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

        function captureFrames() {
            frames = [];
            let captured = 0;
            const total = 20;
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('progressFill').style.width = '0%';
            updateStatus('📸 Capturing frames... Please turn your head slowly left and right.', 'info');
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
                ctx.drawImage(document.getElementById('video'), 0, 0);
                
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

        function verifyWithAPI() {
            updateStatus('🔍 Analyzing your identity and liveness...', 'info');
            document.getElementById('recordingIndicator').classList.remove('active');
            
            if (frames.length < 6) {
                retryCount++;
                updateStatus("❌ Not enough frames captured. Please try again.", "error");
                handleRetryOrManual();
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
                if (data.status === "success" && data.liveness_verified) {
                    userId = data.user_id;
                    updateStatus("✅ Identity verified successfully!", "success");
                    updateStep(3);
                    setTimeout(() => {
                        proceedToPinEntry();
                    }, 1500);
                } else {
                    retryCount++;
                    if (!data.liveness_verified) {
                        if (data.message && data.message.includes("spoof")) {
                            updateStatus("🚫 Spoofing attempt detected. Please try again with a real person.", "error");
                        } else if (data.message && data.message.includes("static")) {
                            updateStatus("🚫 Video replay detected. Live verification required.", "error");
                        } else {
                            updateStatus(`❌ Liveness check failed: ${data.message || "Please follow the instructions carefully."}`, "error");
                        }
                    } else if (data.message && data.message.includes("not recognized")) {
                        updateStatus("⚠️ Face not recognized. Please try again.", "warning");
                    } else {
                        updateStatus(`❌ Verification failed: ${data.message || "Unknown error"}`, "error");
                    }
                    handleRetryOrManual();
                }
            })
            .catch(error => {
                console.error("API error:", error);
                retryCount++;
                updateStatus("❌ Server error during verification. Please try again later.", "error");
                handleRetryOrManual();
            });
        }

        function handleRetryOrManual() {
            if (retryCount >= maxRetries) {
                updateStatus("⚠️ Too many failed attempts. Please try manual verification.", "error");
                goBackToCheckout();
            } else {
                showRetryOption();
            }
        }

        function proceedToPinEntry() {
            // Store user ID and cart data in PHP session via AJAX call
            fetch('../store_session.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    user_id: userId,
                    cart: cart,
                    action: 'store_authenticated_user'
                })
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === "success") {
                    // Also store in sessionStorage as backup
                    sessionStorage.setItem('authenticatedUserId', userId);
                    sessionStorage.setItem('challengeVerified', 'true');
                    
                    // Redirect to the correct PIN entry page
                    window.location.href = 'pin_entry.php';
                } else {
                    updateStatus('Session storage failed. Please try again.', 'error');
                    showRetryOption();
                }
            })
            .catch(error => {
                console.error("Session storage error:", error);
                updateStatus('Session storage failed. Please try again.', 'error');
                showRetryOption();
            });
        }

        function updateStep(step) {
            currentStep = step;
            
            // Reset all steps
            document.querySelectorAll('.scan-step').forEach(s => {
                s.classList.remove('active', 'completed');
            });
            
            // Update steps based on current step
            for (let i = 1; i <= 3; i++) {
                const stepEl = document.getElementById(`step${i}`);
                if (i < step) {
                    stepEl.classList.add('completed');
                } else if (i === step) {
                    stepEl.classList.add('active');
                }
            }
        }

        function updateStatus(message, type = 'info') {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.className = `status-message status-${type}`;
        }

        function showRetryOption() {
            document.getElementById('retryBtn').style.display = 'inline-flex';
        }

        function retryProcess() {
            // Reset everything
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
            
            // Reset UI
            document.getElementById('webcamContainer').style.display = 'none';
            document.getElementById('challengeInstruction').style.display = 'none';
            document.getElementById('challengeInstruction').classList.remove('active');
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('progressFill').style.width = '0%';
            
            // Remove verify button if it exists
            const verifyButton = document.getElementById('verifyButton');
            if (verifyButton) {
                verifyButton.parentElement.remove();
            }
            
            // Reset buttons
            document.getElementById('startScanBtn').style.display = 'inline-flex';
            document.getElementById('retryBtn').style.display = 'none';
            
            // Reset variables
            frames = [];
            faceDetected = false;
            userId = null;
            
            updateStep(1);
            updateStatus('Ready to try again. Click "Start Scan" to begin.', 'info');
        }

        function goBackToCheckout() {
            // Clean up
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            if (faceDetectionInterval) {
                clearInterval(faceDetectionInterval);
            }
            
            window.location.href = 'shopping_cart.php';
        }

        // Clean up on page unload
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