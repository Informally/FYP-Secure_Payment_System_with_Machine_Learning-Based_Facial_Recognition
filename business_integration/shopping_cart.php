<?php
// business_integration/shopping_cart.php
// DEMO VERSION - Shows API integration prominently

// ✅ DEMO: Show the API key configuration prominently
const MERCHANT_ID = 'MCD';
const FACEPAY_API_KEY = 'sk_1f55be80af366a8c3e87f53b14962a4f2ec4bbe14755af39'; // Your actual API key
const FACEPAY_GATEWAY_URL = 'http://localhost/FYP/gateway/checkout.php';
const SUCCESS_URL = 'http://localhost/FYP/business_integration/payment_success.php';
const CANCEL_URL = 'http://localhost/FYP/business_integration/payment_failed.php';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkout - Prototype Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* Your existing styles plus demo highlights */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
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

        /* ✅ DEMO: API Integration Showcase Panel */
        .api-showcase {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            margin: 20px auto;
            max-width: 1200px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            animation: pulse-glow 3s infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
            50% { box-shadow: 0 8px 35px rgba(102, 126, 234, 0.6); }
        }

        .api-showcase h3 {
            font-size: 24px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .api-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }

        .api-detail {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .api-detail strong {
            color: #ffc72c;
        }

        .api-key-display {
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 5px;
            word-break: break-all;
            margin-top: 5px;
            border: 1px solid #ffc72c;
        }

        .checkout-container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            flex: 1;
        }

        .section-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 3px solid #ffc72c;
        }

        .section-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            color: white;
            padding: 20px;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-content {
            padding: 25px;
        }

        /* Payment Gateway Demo Section */
        .gateway-demo {
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            border: 3px solid #4caf50;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .gateway-demo h4 {
            color: #2e7d32;
            margin-bottom: 15px;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .demo-step {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 4px solid #4caf50;
            font-size: 14px;
        }

        .payment-option {
            border: 3px solid #ecf0f1;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            margin-bottom: 20px;
        }

        .payment-option:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(39, 174, 96, 0.3);
        }

        .payment-icon {
            font-size: 48px;
            margin-bottom: 15px;
            display: block;
        }

        .payment-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .payment-description {
            font-size: 14px;
            color: rgba(255,255,255,0.9);
            line-height: 1.4;
        }

        .action-buttons {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }

        .btn {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #95a5a6;
            color: white;
        }

        /* Demo Console */
        .demo-console {
            background: #1e1e1e;
            color: #00ff00;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            margin-top: 20px;
            border: 2px solid #333;
            max-height: 200px;
            overflow-y: auto;
        }

        .console-line {
            margin-bottom: 5px;
        }

        .console-prompt {
            color: #ffc72c;
        }

        /* Order Summary Styles */
        .order-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 0;
            border-bottom: 1px solid #eee;
        }

        .item-image {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #ffc72c, #ffeb3b);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: #da291c;
        }

        .item-details {
            flex: 1;
        }

        .item-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            font-size: 16px;
        }

        .item-price {
            color: #da291c;
            font-weight: bold;
            font-size: 16px;
        }

        .item-quantity {
            background: #da291c;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }

        .total-section {
            background: linear-gradient(135deg, #34495e, #2c3e50);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .total-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 16px;
        }

        .total-row.final {
            font-size: 24px;
            font-weight: bold;
            border-top: 2px solid rgba(255,255,255,0.3);
            padding-top: 15px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-credit-card"></i> Prototype Kiosk</h1>
    </div>

    <!-- ✅ DEMO: API Integration Showcase
    <div class="api-showcase">
        <h3><i class="fas fa-code"></i> FacePay Payment Gateway Integration</h3>
        <p><strong>🚀 This kiosk integrates with FacePay as an external merchant (like McDonald's using PayPal)</strong></p>
        
        <div class="api-details">
            <div class="api-detail">
                <strong>Merchant ID:</strong> <?= MERCHANT_ID ?>
                <div>Registered in FacePay merchant system</div>
            </div>
            <div class="api-detail">
                <strong>API Key:</strong>
                <div class="api-key-display"><?= FACEPAY_API_KEY ?></div>
            </div>
            <div class="api-detail">
                <strong>Gateway URL:</strong> <?= FACEPAY_GATEWAY_URL ?>
                <div>Payment processing endpoint</div>
            </div>
            <div class="api-detail">
                <strong>Integration Type:</strong> Gateway Redirect
                <div>Like PayPal/Stripe checkout flow</div>
            </div>
        </div>
    </div> -->

    <div class="checkout-container">
        <!-- Order Summary -->
        <div class="section-card">
            <div class="section-header">
                <i class="fas fa-receipt"></i>
                Your Order Summary
            </div>
            <div class="section-content">
                <div id="orderItems"></div>
                <div class="total-section">
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

        <!-- Payment Gateway Demo -->
        <div class="section-card">
            <div class="section-header">
                <i class="fas fa-credit-card"></i>
                Payment Options
            </div>

                <div class="payment-option">
                    <i class="payment-icon fas fa-user-check"></i>
                    <div class="payment-title">FacePay Gateway</div>
                    <div class="payment-description">
                        External payment processing via API integration<br>
                        <strong>Merchant:</strong> <?= MERCHANT_ID ?><br>
                        <strong>API:</strong> ...<?= substr(FACEPAY_API_KEY, -8) ?>
                    </div>
                </div>

                <!-- Hidden gateway form -->
                <form id="facepayGatewayForm" method="POST" action="<?= FACEPAY_GATEWAY_URL ?>" style="display: none;">
                    <input type="hidden" name="merchant_id" value="<?= MERCHANT_ID ?>">
                    <input type="hidden" name="api_key" value="<?= FACEPAY_API_KEY ?>">
                    <input type="hidden" name="amount" id="gateway_amount">
                    <input type="hidden" name="order_id" id="gateway_order_id">
                    <input type="hidden" name="currency" value="MYR">
                    <input type="hidden" name="return_url" value="<?= SUCCESS_URL ?>">
                    <input type="hidden" name="cancel_url" value="<?= CANCEL_URL ?>">
                    <input type="hidden" name="description" id="gateway_description">
                </form>

                <div class="action-buttons">
                    <button class="btn btn-secondary" onclick="goBack()">
                        <i class="fas fa-arrow-left"></i> Back to Menu
                    </button>
                    <button class="btn btn-primary" onclick="payWithGateway()">
                        <i class="fas fa-rocket"></i> Initiate Gateway Payment
                    </button>
                </div>

                <!-- ✅ DEMO: Live API Console -->
                <div class="demo-console" id="apiConsole">
                    <div class="console-line"><span class="console-prompt">$</span> Initializing FacePay Gateway Integration...</div>
                    <div class="console-line">✅ Merchant ID: <?= MERCHANT_ID ?></div>
                    <div class="console-line">✅ API Key: <?= substr(FACEPAY_API_KEY, 0, 20) ?>...</div>
                    <div class="console-line">✅ Gateway endpoint ready</div>
                    <div class="console-line"><span class="console-prompt">$</span> Waiting for payment initiation...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let cart = [];

        document.addEventListener('DOMContentLoaded', function() {
            loadCartFromStorage();
            renderOrderSummary();
            startDemoAnimation();
        });

        function loadCartFromStorage() {
            const savedCart = sessionStorage.getItem('kioskCart');
            if (savedCart) {
                cart = JSON.parse(savedCart);
            }
            if (cart.length === 0) {
                // Add demo items for presentation
                cart = [
                    {id: 1, name: "Big Mac", price: 15.90, quantity: 1, icon: "🍔"},
                    {id: 2, name: "Large Fries", price: 6.50, quantity: 1, icon: "🍟"},
                    {id: 3, name: "Coca-Cola", price: 4.90, quantity: 1, icon: "🥤"}
                ];
            }
        }

        function renderOrderSummary() {
            const orderItems = document.getElementById('orderItems');
            const subtotalEl = document.getElementById('subtotal');
            const taxEl = document.getElementById('tax'); 
            const finalTotalEl = document.getElementById('finalTotal');

            orderItems.innerHTML = cart.map(item => `
                <div class="order-item">
                    <div class="item-image">${item.icon}</div>
                    <div class="item-details">
                        <div class="item-name">${item.name}</div>
                        <div class="item-price">RM ${item.price.toFixed(2)}</div>
                    </div>
                    <div class="item-quantity">Qty: ${item.quantity}</div>
                </div>
            `).join('');

            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.06;
            const finalTotal = subtotal + tax;

            subtotalEl.textContent = subtotal.toFixed(2);
            taxEl.textContent = tax.toFixed(2);
            finalTotalEl.textContent = finalTotal.toFixed(2);
        }

        // ✅ DEMO: Gateway payment with console logging
        function payWithGateway() {
            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.06;
            const total = subtotal + tax;
            const orderId = 'KIOSK-DEMO-' + Date.now();
            
            // ✅ DEMO: Show API call in console
            addConsoleLog('🚀 Initiating gateway payment...');
            addConsoleLog(`💰 Amount: RM ${total.toFixed(2)}`);
            addConsoleLog(`📋 Order ID: ${orderId}`);
            addConsoleLog(`🔑 Using API Key: ${<?= json_encode(FACEPAY_API_KEY) ?>}`);
            addConsoleLog('🌐 Redirecting to FacePay Gateway...');
            
            // Fill form data
            document.getElementById('gateway_amount').value = total.toFixed(2);
            document.getElementById('gateway_order_id').value = orderId;
            document.getElementById('gateway_description').value = `Demo Kiosk Order - ${cart.length} items`;
            
            // ✅ DEMO: Short delay to show the console, then redirect
            setTimeout(() => {
                document.getElementById('facepayGatewayForm').submit();
            }, 2000);
        }

        function addConsoleLog(message) {
            const console = document.getElementById('apiConsole');
            const line = document.createElement('div');
            line.className = 'console-line';
            line.innerHTML = `<span class="console-prompt">$</span> ${message}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }

        function startDemoAnimation() {
            // ✅ DEMO: Add some live API status updates
            setTimeout(() => addConsoleLog('✅ Gateway connection established'), 2000);
            setTimeout(() => addConsoleLog('✅ Merchant authentication successful'), 4000);
            setTimeout(() => addConsoleLog('🎯 Ready for payment processing'), 6000);
        }

        function goBack() {
            window.location.href = 'e-commerce.php';
        }
    </script>
</body>
</html>