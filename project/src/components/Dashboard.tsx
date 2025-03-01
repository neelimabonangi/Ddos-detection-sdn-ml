import React from 'react';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { AlertTriangle, Shield, Activity } from 'lucide-react';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

const Dashboard = () => {
  // Sample data for demonstration
  const detectionData = {
    labels: ['Normal', 'DDoS Attack'],
    datasets: [
      {
        label: 'Traffic Classification',
        data: [65, 35],
        backgroundColor: ['rgba(54, 162, 235, 0.6)', 'rgba(255, 99, 132, 0.6)'],
        borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const modelAccuracyData = {
    labels: ['Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Naive Bayes'],
    datasets: [
      {
        label: 'Accuracy (%)',
        data: [99.88, 97.5, 96.2, 98.1, 94.3],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(255, 205, 86, 0.6)',
          'rgba(201, 203, 207, 0.6)',
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(201, 203, 207, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">DDoS Detection Dashboard</h2>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Traffic Analyzed</p>
              <p className="text-2xl font-bold">104,345</p>
            </div>
            <Activity className="text-blue-500" size={24} />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">DDoS Attacks Detected</p>
              <p className="text-2xl font-bold">36,521</p>
            </div>
            <AlertTriangle className="text-red-500" size={24} />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Current Model Accuracy</p>
              <p className="text-2xl font-bold">99.88%</p>
            </div>
            <Shield className="text-green-500" size={24} />
          </div>
        </div>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Traffic Classification</h3>
          <div className="h-64">
            <Pie data={detectionData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Model Accuracy Comparison</h3>
          <div className="h-64">
            <Bar 
              data={modelAccuracyData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100
                  }
                }
              }} 
            />
          </div>
        </div>
      </div>
      
      {/* Recent Alerts */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Recent DDoS Alerts</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source IP</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Target</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Attack Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2025-06-10 14:23:15</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">192.168.1.45</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Web Server</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">SYN Flood</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">99.7%</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Blocked</span>
                </td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2025-06-10 14:15:32</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">10.45.67.89</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">DNS Server</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">UDP Flood</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">98.2%</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Blocked</span>
                </td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2025-06-10 13:58:47</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">172.16.23.12</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">API Gateway</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">HTTP Flood</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">97.5%</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Blocked</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;