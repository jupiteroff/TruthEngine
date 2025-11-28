document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const symbol = document.getElementById('symbol').value;
    const target = document.getElementById('target').value;
    const btn = document.getElementById('submitBtn');
    const resultSection = document.getElementById('resultSection');

    // UI Loading State
    btn.classList.add('loading');
    btn.disabled = true;
    resultSection.classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, target })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Populate Results
        document.getElementById('resSymbol').textContent = data.symbol;
        document.getElementById('resTargetTime').textContent = `Target: ${data.target_dt}`;
        document.getElementById('resLastPrice').textContent = formatPrice(data.last_price);
        document.getElementById('resForecast').textContent = formatPrice(data.forecast_price);

        const pctEl = document.getElementById('resPct');
        pctEl.textContent = `${data.pct_change > 0 ? '+' : ''}${data.pct_change.toFixed(3)}%`;
        pctEl.className = `pct ${data.pct_change >= 0 ? 'positive' : 'negative'}`;

        document.getElementById('resTrainCCC').textContent = data.train_ccc.toFixed(4);
        document.getElementById('resTestCCC').textContent = data.test_ccc.toFixed(4);
        document.getElementById('resAlpha').textContent = data.alpha.toFixed(4);
        document.getElementById('resInterval').textContent = data.interval;

        // Populate Factors
        const grid = document.getElementById('factorsGrid');
        grid.innerHTML = '';

        // Sort betas by absolute magnitude to show most important first
        const sortedBetas = Object.entries(data.betas)
            .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a));

        sortedBetas.forEach(([name, val]) => {
            const div = document.createElement('div');
            div.className = 'factor-item';
            div.innerHTML = `
                <span class="factor-name">${name}</span>
                <span class="factor-val" style="color: ${val >= 0 ? '#00ff88' : '#ff0055'}">
                    ${val.toFixed(4)}
                </span>
            `;
            grid.appendChild(div);
        });

        // Show Results
        resultSection.classList.remove('hidden');

    } catch (err) {
        console.error(err);
        alert('Failed to fetch prediction. Check console.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    }).format(price);
}

// Live Price Ticker
async function fetchLivePrices() {
    try {
        const response = await fetch('/api/live-prices');
        const data = await response.json();

        if (data.error) {
            console.error('Error fetching prices:', data.error);
            return;
        }

        const container = document.getElementById('priceTicker');
        container.innerHTML = '';

        // Create ticker items for each crypto
        Object.entries(data).forEach(([symbol, info]) => {
            const item = document.createElement('div');
            item.className = 'ticker-item';

            const changeClass = info.change_24h >= 0 ? 'positive' : 'negative';
            const changeSign = info.change_24h >= 0 ? '+' : '';

            item.innerHTML = `
                <div class="ticker-symbol">${symbol}</div>
                <div class="ticker-price">$${formatPrice(info.price)}</div>
                <div class="ticker-change ${changeClass}">${changeSign}${info.change_24h.toFixed(2)}%</div>
            `;

            container.appendChild(item);
        });

    } catch (error) {
        console.error('Failed to fetch live prices:', error);
    }
}

// Fetch prices on page load
fetchLivePrices();

// Auto-refresh prices every 10 seconds
setInterval(fetchLivePrices, 10000);
