<?php
require_once '../customer_side/config.php';
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prototype Kiosk - Self Service</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            background: #fff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* McDonald's style header */
        .kiosk-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            padding: 15px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .kiosk-header h1 {
            color: white;
            font-size: 2.8rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 5px;
        }

        .kiosk-header p {
            color: white;
            font-size: 1.3rem;
            font-weight: 500;
        }

        .main-container {
            flex: 1;
            display: flex;
            background: #f8f8f8;
        }

        /* Left sidebar for categories */
        .categories-sidebar {
            width: 250px;
            background: white;
            border-right: 3px solid #da291c;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }

        .category-header {
            background: #da291c;
            color: white;
            padding: 20px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }

        .category-item {
            display: flex;
            align-items: center;
            padding: 20px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
            transition: all 0.3s ease;
            font-size: 16px;
            font-weight: 500;
        }

        .category-item:hover {
            background: #fff3cd;
            border-left: 5px solid #ffc72c;
        }

        .category-item.active {
            background: #ffc72c;
            color: #da291c;
            font-weight: bold;
            border-left: 5px solid #da291c;
        }

        .category-icon {
            margin-right: 15px;
            font-size: 24px;
            width: 30px;
            text-align: center;
        }

        /* Main content area */
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .products-section {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            max-height: calc(100vh - 200px);
        }

        .section-title {
            font-size: 28px;
            color: #da291c;
            margin-bottom: 25px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
        }

        .product-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            border: 3px solid transparent;
        }

        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            border-color: #ffc72c;
        }

        .product-image {
            height: 180px;
            background: linear-gradient(135deg, #ffc72c, #ffeb3b);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 72px;
            color: #da291c;
            position: relative;
        }

        .product-details {
            padding: 20px;
        }

        .product-name {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            line-height: 1.3;
        }

        .product-description {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            line-height: 1.4;
        }

        .product-price {
            font-size: 24px;
            font-weight: bold;
            color: #da291c;
            margin-bottom: 15px;
        }

        .add-to-order-btn {
            background: linear-gradient(45deg, #da291c, #ff4444);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
        }

        .add-to-order-btn:hover {
            background: linear-gradient(45deg, #c41e3a, #da291c);
            transform: scale(1.02);
        }

        /* Cart section - fixed at bottom */
        .cart-section {
            background: white;
            border-top: 4px solid #da291c;
            padding: 20px 30px;
            box-shadow: 0 -4px 15px rgba(0,0,0,0.1);
            position: sticky;
            bottom: 0;
        }

        .cart-summary {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .cart-info {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .cart-items-count {
            background: #da291c;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 16px;
        }

        .cart-total {
            font-size: 24px;
            font-weight: bold;
            color: #da291c;
        }

        .cart-actions {
            display: flex;
            gap: 15px;
        }

        .view-order-btn {
            background: #ffc72c;
            color: #da291c;
            border: 2px solid #da291c;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .view-order-btn:hover {
            background: #da291c;
            color: white;
        }

        .checkout-btn {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }

        .checkout-btn:hover {
            background: linear-gradient(45deg, #2ecc71, #27ae60);
            transform: scale(1.05);
        }

        .checkout-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }

        /* Order details modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
        }

        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 0;
            border-radius: 15px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .modal-header {
            background: linear-gradient(45deg, #da291c, #ffc72c);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h2 {
            font-size: 24px;
            font-weight: bold;
        }

        .close-btn {
            background: none;
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 5px;
        }

        .modal-body {
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }

        .order-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 0;
            border-bottom: 1px solid #eee;
        }

        .order-item:last-child {
            border-bottom: none;
        }

        .item-image-small {
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
        }

        .item-price {
            color: #da291c;
            font-weight: bold;
        }

        .quantity-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #f8f9fa;
            border-radius: 20px;
            padding: 5px;
        }

        .quantity-btn {
            background: #da291c;
            color: white;
            border: none;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .quantity-btn:hover {
            background: #c41e3a;
        }

        .quantity {
            font-weight: bold;
            min-width: 30px;
            text-align: center;
        }

        .remove-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            margin-left: 10px;
        }

        .modal-footer {
            background: #f8f9fa;
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-total {
            font-size: 20px;
            font-weight: bold;
            color: #da291c;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #7f8c8d;
        }

        .empty-state i {
            font-size: 64px;
            margin-bottom: 20px;
            color: #bdc3c7;
        }

        /* Responsive design */
        @media (max-width: 1024px) {
            .main-container {
                flex-direction: column;
            }

            .categories-sidebar {
                width: 100%;
                display: flex;
                overflow-x: auto;
                border-right: none;
                border-bottom: 3px solid #da291c;
            }

            .category-header {
                display: none;
            }

            .category-item {
                min-width: 150px;
                justify-content: center;
                border-bottom: none;
                border-right: 1px solid #eee;
            }

            .products-grid {
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }
        }

        @media (max-width: 768px) {
            .kiosk-header h1 {
                font-size: 2rem;
            }

            .products-grid {
                grid-template-columns: 1fr;
            }

            .cart-summary {
                flex-direction: column;
                gap: 15px;
            }

            .cart-actions {
                width: 100%;
                justify-content: space-between;
            }
        }

        /* Notification */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #27ae60;
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            z-index: 1001;
            transform: translateX(400px);
            transition: transform 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.error {
            background: #e74c3c;
        }
    </style>
</head>
<body>
    <div class="kiosk-header">
        <h1><i class="fas fa-store"></i> Prototype Kiosk</h1>
        <p>Touch to Start Your Order</p>
    </div>

    <div class="main-container">
        <!-- Categories Sidebar -->
        <div class="categories-sidebar">
            <div class="category-header">
                <i class="fas fa-list"></i> Menu
            </div>
            <div class="category-item active" data-category="all">
                <span class="category-icon">🍽️</span>
                <span>All Menu</span>
            </div>
            <div class="category-item" data-category="Burgers">
                <span class="category-icon">🍔</span>
                <span>Burgers</span>
            </div>
            <div class="category-item" data-category="Chicken">
                <span class="category-icon">🍗</span>
                <span>Chicken</span>
            </div>
            <div class="category-item" data-category="Sides">
                <span class="category-icon">🍟</span>
                <span>Sides</span>
            </div>
            <div class="category-item" data-category="Desserts">
                <span class="category-icon">🍦</span>
                <span>Desserts</span>
            </div>
            <div class="category-item" data-category="Drinks">
                <span class="category-icon">🥤</span>
                <span>Drinks</span>
            </div>
        </div>

        <!-- Content Area -->
        <div class="content-area">
            <!-- Products Section -->
            <div class="products-section">
                <div class="section-title">
                    <i class="fas fa-utensils"></i>
                    <span id="categoryTitle">All Menu Items</span>
                </div>

                <div class="products-grid" id="productsGrid">
                    <!-- Products will be loaded here -->
                </div>
            </div>

            <!-- Cart Section -->
            <div class="cart-section">
                <div class="cart-summary">
                    <div class="cart-info">
                        <div class="cart-items-count">
                            <i class="fas fa-shopping-cart"></i>
                            <span id="cartCount">0</span> Items
                        </div>
                        <div class="cart-total">
                            Total: RM <span id="totalAmount">0.00</span>
                        </div>
                    </div>
                    <div class="cart-actions">
                        <button class="view-order-btn" onclick="showOrderDetails()">
                            <i class="fas fa-list"></i> View Order
                        </button>
                        <button class="checkout-btn" id="checkoutBtn" disabled>
                            <i class="fas fa-credit-card"></i> Checkout
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Order Details Modal -->
    <div id="orderModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-receipt"></i> Your Order</h2>
                <button class="close-btn" onclick="closeOrderModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
                <!-- Order items will be loaded here -->
            </div>
            <div class="modal-footer">
                <div class="modal-total">
                    Total: RM <span id="modalTotal">0.00</span>
                </div>
                <button class="checkout-btn" onclick="proceedToCheckout()">
                    <i class="fas fa-arrow-right"></i> Continue to Payment
                </button>
            </div>
        </div>
    </div>

    <!-- Notification -->
    <div class="notification" id="notification"></div>

    <script>
        // Sample food products data
        const products = [
            {
                id: 1,
                name: "Big Mac",
                description: "Two beef patties, lettuce, cheese",
                price: 15.90,
                category: "Burgers",
                icon: "🍔",
                stock: 50
            },
            {
                id: 2,
                name: "Quarter Pounder",
                description: "Quarter pound of fresh beef with cheese",
                price: 18.50,
                category: "Burgers",
                icon: "🍔",
                stock: 45
            },
            {
                id: 3,
                name: "McChicken",
                description: "Crispy chicken fillet with lettuce and mayo",
                price: 12.90,
                category: "Burgers",
                icon: "🍗",
                stock: 60
            },
            {
                id: 4,
                name: "Chicken McNuggets (6pc)",
                description: "Six pieces of crispy chicken nuggets",
                price: 8.90,
                category: "Chicken",
                icon: "🍗",
                stock: 80
            },
            {
                id: 5,
                name: "Chicken McNuggets (10pc)",
                description: "Ten pieces of crispy chicken nuggets",
                price: 13.90,
                category: "Chicken",
                icon: "🍗",
                stock: 75
            },
            {
                id: 6,
                name: "Spicy Chicken Deluxe",
                description: "Spicy crispy chicken with fresh lettuce",
                price: 16.90,
                category: "Chicken",
                icon: "🌶️",
                stock: 40
            },
            {
                id: 7,
                name: "Large Fries",
                description: "Golden crispy french fries",
                price: 6.50,
                category: "Sides",
                icon: "🍟",
                stock: 100
            },
            {
                id: 8,
                name: "Medium Fries",
                description: "Golden crispy french fries",
                price: 5.50,
                category: "Sides",
                icon: "🍟",
                stock: 100
            },
            {
                id: 9,
                name: "Apple Pie",
                description: "Hot apple pie with cinnamon",
                price: 4.50,
                category: "Desserts",
                icon: "🥧",
                stock: 60
            },
            {
                id: 10,
                name: "McFlurry Oreo",
                description: "Vanilla soft serve with Oreo cookies",
                price: 7.90,
                category: "Desserts",
                icon: "🍦",
                stock: 45
            },
            {
                id: 11,
                name: "Chocolate Chip Cookie",
                description: "Freshly baked chocolate chip cookie",
                price: 3.50,
                category: "Desserts",
                icon: "🍪",
                stock: 80
            },
            {
                id: 12,
                name: "Coca-Cola (Large)",
                description: "Large refreshing Coca-Cola",
                price: 4.90,
                category: "Drinks",
                icon: "🥤",
                stock: 100
            },
            {
                id: 13,
                name: "Coca-Cola (Medium)",
                description: "Medium refreshing Coca-Cola",
                price: 3.90,
                category: "Drinks",
                icon: "🥤",
                stock: 100
            },
            {
                id: 14,
                name: "Orange Juice",
                description: "Fresh orange juice",
                price: 5.50,
                category: "Drinks",
                icon: "🧃",
                stock: 70
            },
            {
                id: 15,
                name: "Hot Coffee",
                description: "Freshly brewed coffee",
                price: 4.50,
                category: "Drinks",
                icon: "☕",
                stock: 90
            },
            {
                id: 16,
                name: "Iced Coffee",
                description: "Refreshing iced coffee",
                price: 5.50,
                category: "Drinks",
                icon: "🧊",
                stock: 85
            }
        ];

        let cart = [];
        let currentCategory = 'all';

        // Initialize the kiosk
        document.addEventListener('DOMContentLoaded', function() {
            loadCartFromStorage();
            renderProducts();
            setupEventListeners();
            updateCartDisplay();
        });

        function setupEventListeners() {
            // Category selection
            document.querySelectorAll('.category-item').forEach(item => {
                item.addEventListener('click', function() {
                    document.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
                    this.classList.add('active');
                    currentCategory = this.dataset.category;
                    updateCategoryTitle();
                    renderProducts();
                });
            });

            // Checkout button
            document.getElementById('checkoutBtn').addEventListener('click', function() {
                if (cart.length > 0) {
                    proceedToCheckout();
                }
            });

            // Modal close on outside click
            document.getElementById('orderModal').addEventListener('click', function(e) {
                if (e.target === this) {
                    closeOrderModal();
                }
            });
        }

        function updateCategoryTitle() {
            const titles = {
                'all': 'All Menu Items',
                'Burgers': 'Burgers',
                'Chicken': 'Chicken & More',
                'Sides': 'Sides',
                'Desserts': 'Desserts',
                'Drinks': 'Beverages'
            };
            document.getElementById('categoryTitle').textContent = titles[currentCategory] || 'All Menu Items';
        }

        function renderProducts() {
            const grid = document.getElementById('productsGrid');
            const filteredProducts = currentCategory === 'all' 
                ? products 
                : products.filter(p => p.category === currentCategory);

            if (filteredProducts.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-box-open"></i>
                        <h3>No items found</h3>
                        <p>Try selecting a different category</p>
                    </div>
                `;
                return;
            }

            grid.innerHTML = filteredProducts.map(product => `
                <div class="product-card" onclick="addToCart(${product.id})">
                    <div class="product-image">
                        ${product.icon}
                    </div>
                    <div class="product-details">
                        <div class="product-name">${product.name}</div>
                        <div class="product-description">${product.description}</div>
                        <div class="product-price">RM ${product.price.toFixed(2)}</div>
                        <button class="add-to-order-btn" onclick="event.stopPropagation(); addToCart(${product.id})">
                            <i class="fas fa-plus"></i> Add to Order
                        </button>
                    </div>
                </div>
            `).join('');
        }

        function addToCart(productId) {
            const product = products.find(p => p.id === productId);
            if (!product) return;

            const existingItem = cart.find(item => item.id === productId);
            
            if (existingItem) {
                existingItem.quantity += 1;
            } else {
                cart.push({
                    ...product,
                    quantity: 1
                });
            }

            updateCartDisplay();
            saveCartToStorage();
            showNotification(`${product.name} added to order!`);
        }

        function removeFromCart(productId) {
            cart = cart.filter(item => item.id !== productId);
            updateCartDisplay();
            saveCartToStorage();
            showNotification('Item removed from order', 'error');
        }

        function updateQuantity(productId, change) {
            const item = cart.find(item => item.id === productId);
            if (!item) return;

            item.quantity += change;
            
            if (item.quantity <= 0) {
                removeFromCart(productId);
                return;
            }

            updateCartDisplay();
            saveCartToStorage();
        }

        function updateCartDisplay() {
            const cartCount = document.getElementById('cartCount');
            const totalAmount = document.getElementById('totalAmount');
            const checkoutBtn = document.getElementById('checkoutBtn');

            // Update cart count
            const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);
            cartCount.textContent = totalItems;

            // Calculate and display total
            const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            totalAmount.textContent = total.toFixed(2);

            // Enable/disable checkout button
            checkoutBtn.disabled = cart.length === 0;
        }

        function showOrderDetails() {
            const modal = document.getElementById('orderModal');
            const modalBody = document.getElementById('modalBody');
            const modalTotal = document.getElementById('modalTotal');

            if (cart.length === 0) {
                modalBody.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-shopping-cart"></i>
                        <h3>Your order is empty</h3>
                        <p>Add some items to get started</p>
                    </div>
                `;
            } else {
                modalBody.innerHTML = cart.map(item => `
                    <div class="order-item">
                        <div class="item-image-small">${item.icon}</div>
                        <div class="item-details">
                            <div class="item-name">${item.name}</div>
                            <div class="item-price">RM ${item.price.toFixed(2)}</div>
                        </div>
                        <div class="quantity-controls">
                            <button class="quantity-btn" onclick="updateQuantity(${item.id}, -1)">
                                <i class="fas fa-minus"></i>
                            </button>
                            <div class="quantity">${item.quantity}</div>
                            <button class="quantity-btn" onclick="updateQuantity(${item.id}, 1)">
                                <i class="fas fa-plus"></i>
                            </button>
                        </div>
                        <button class="remove-btn" onclick="removeFromCart(${item.id})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `).join('');
            }

            // Update modal total
            const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            modalTotal.textContent = total.toFixed(2);

            modal.style.display = 'block';
        }

        function closeOrderModal() {
            document.getElementById('orderModal').style.display = 'none';
        }

        function proceedToCheckout() {
            if (cart.length === 0) {
                showNotification('Please add items to your order first', 'error');
                return;
            }
            
            saveCartToStorage();
            window.location.href = 'shopping_cart.php';
        }

        function saveCartToStorage() {
            sessionStorage.setItem('kioskCart', JSON.stringify(cart));
        }

        function loadCartFromStorage() {
            const savedCart = sessionStorage.getItem('kioskCart');
            if (savedCart) {
                cart = JSON.parse(savedCart);
            }
        }

        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type}`;
            notification.classList.add('show');

            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
    </script>
</body>
</html>