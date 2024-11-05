import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    LineElement,
    PointElement,
    LinearScale,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

// Register the components to be used in the chart
ChartJS.register(LineElement, PointElement, LinearScale, Title, Tooltip, Legend);

function App() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [chartData, setChartData] = useState({});

    // Fetch data from the Flask API on component mount
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/api/data');
                setData(response.data);
                setLoading(false);

                // Prepare data for yearly average prices
                const yearlyPrices = {};
                response.data.forEach(item => {
                    const year = new Date(item.Date).getFullYear();
                    if (!yearlyPrices[year]) {
                        yearlyPrices[year] = { total: 0, count: 0 };
                    }
                    yearlyPrices[year].total += item.Price;
                    yearlyPrices[year].count += 1;
                });

                // Create labels and data for the chart
                const labels = Object.keys(yearlyPrices);
                const prices = labels.map(year => yearlyPrices[year].total / yearlyPrices[year].count);

                setChartData({
                    labels: labels,
                    datasets: [
                        {
                            label: 'Average Brent Oil Prices (Yearly)',
                            data: prices,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderWidth: 1,
                        },
                    ],
                });
            } catch (err) {
                setError(err);
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    // Key events that influenced oil prices
    // Key events that influenced oil prices
const keyEvents = [
  { year: 1987, event: "Black Monday; stock market crash influences oil prices." },
  { year: 1990, event: "Gulf War leads to a spike in oil prices." },
  { year: 1991, event: "Post-Gulf War adjustments; oil prices stabilize." },
  { year: 2001, event: "September 11 attacks; oil prices rise." },
  { year: 2003, event: "Iraq War begins, causing volatility in oil markets." },
  { year: 2008, event: "Global financial crisis causes oil prices to plummet." },
  { year: 2010, event: "Deepwater Horizon oil spill impacts market perception." },
  { year: 2014, event: "Oil prices decline due to oversupply and OPEC decisions." },
  { year: 2016, event: "OPEC cuts production to stabilize falling prices." },
  { year: 2019, event: "Drone attacks on Saudi oil facilities cause price spikes." },
  { year: 2020, event: "COVID-19 pandemic leads to unprecedented drop in demand." },
  { year: 2021, event: "OPEC+ agreements lead to gradual price recovery." },
];


    // External factors affecting oil prices over the years
    const externalFactors = [
        { year: 1987, inflation: 3.66, gdp: 3.4, exchangeRate: 1.66, tradeBalance: -23.5 },
        { year: 1990, inflation: 5.40, gdp: 1.9, exchangeRate: 1.60, tradeBalance: -18.7 },
        { year: 2000, inflation: 3.38, gdp: 4.1, exchangeRate: 1.06, tradeBalance: -25.1 },
        { year: 2008, inflation: 3.84, gdp: 0.0, exchangeRate: 1.50, tradeBalance: -28.7 },
        { year: 2014, inflation: 1.62, gdp: 2.5, exchangeRate: 1.25, tradeBalance: -25.5 },
        { year: 2020, inflation: 1.23, gdp: -3.4, exchangeRate: 1.27, tradeBalance: -37.3 },
    ];

    // Loading and error handling
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error fetching data: {error.message}</div>;

    // Render the dashboard
    return (
        <div style={{ padding: '20px' }}>
            <h1>Brent Oil Prices Dashboard</h1>
            <p>This dashboard displays the average historical prices of Brent crude oil per year from 1987 to 2020, along with key events and external factors that influenced the oil market.</p>
            
            <h2>Key Events (1987 - 2020)</h2>
            <ul>
                {keyEvents.map((event, index) => (
                    <li key={index}>{event.year}: {event.event}</li>
                ))}
            </ul>

            <h2>External Factors (Inflation, GDP, Exchange Rate, Trade Balance)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Year</th>
                        <th>Inflation (%)</th>
                        <th>GDP Growth (%)</th>
                        <th>Exchange Rate (USD to Local Currency)</th>
                        <th>Trade Balance (Billion USD)</th>
                    </tr>
                </thead>
                <tbody>
                    {externalFactors.map((factor, index) => (
                        <tr key={index}>
                            <td>{factor.year}</td>
                            <td>{factor.inflation}</td>
                            <td>{factor.gdp}</td>
                            <td>{factor.exchangeRate}</td>
                            <td>{factor.tradeBalance}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <h2>Average Price Chart</h2>
            <Line 
                data={chartData} 
                options={{ responsive: true, scales: { y: { beginAtZero: true } } }} 
            />

            <h2>Brent Oil Prices Data (Showing first 10 entries)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
                    {data.slice(0, 10).map((item, index) => ( // Show only the first 10 entries
                        <tr key={index}>
                            <td>{item.Date}</td>
                            <td>{item.Price}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

export default App;
