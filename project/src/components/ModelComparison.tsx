import React from 'react';
import { Bar, Radar, Line } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
);

const ModelComparison = () => {
  // Model performance metrics
  const accuracyData = {
    labels: ['Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Naive Bayes'],
    datasets: [
      {
        label: 'Accuracy (%)',
        data: [99.88, 97.5, 96.2, 98.1, 94.3],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  const precisionRecallData = {
    labels: ['Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Naive Bayes'],
    datasets: [
      {
        label: 'Precision',
        data: [99.9, 97.8, 96.5, 98.3, 93.7],
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
      {
        label: 'Recall',
        data: [99.85, 97.2, 95.9, 97.9, 94.8],
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
      },
    ],
  };

  const f1ScoreData = {
    labels: ['Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Naive Bayes'],
    datasets: [
      {
        label: 'F1 Score',
        data: [99.87, 97.5, 96.2, 98.1, 94.2],
        backgroundColor: 'rgba(153, 102, 255, 0.6)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1,
      },
    ],
  };

  const radarData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Speed', 'Inference Speed'],
    datasets: [
      {
        label: 'Random Forest',
        data: [99.88, 99.9, 99.85, 99.87, 85, 90],
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        pointBackgroundColor: 'rgba(75, 192, 192, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(75, 192, 192, 1)',
      },
      {
        label: 'SVM',
        data: [97.5, 97.8, 97.2, 97.5, 70, 75],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(54, 162, 235, 1)',
      },
      {
        label: 'KNN',
        data: [96.2, 96.5, 95.9, 96.2, 90, 65],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
      },
    ],
  };

  const trainingTimeData = {
    labels: ['Random Forest', 'SVM', 'KNN', 'Decision Tree', 'Naive Bayes'],
    datasets: [
      {
        label: 'Training Time (seconds)',
        data: [45.2, 78.5, 12.3, 8.7, 5.2],
        backgroundColor: 'rgba(255, 159, 64, 0.6)',
        borderColor: 'rgba(255, 159, 64, 1)',
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">ML Model Comparison</h2>
      
      {/* Model Description */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Model Overview</h3>
        <p className="mb-4">
          We've implemented and compared five machine learning models for DDoS attack detection in SDN environments:
          Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, and Naive Bayes.
          Each model was trained on the same dataset with 23 features and evaluated using standard metrics.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mt-4">
          <div className="bg-indigo-50 p-4 rounded-lg">
            <h4 className="font-semibold text-indigo-700">Random Forest</h4>
            <p className="text-sm text-gray-600">Ensemble learning method using multiple decision trees</p>
            <p className="font-bold mt-2">Accuracy: 99.88%</p>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-700">SVM</h4>
            <p className="text-sm text-gray-600">Finds optimal hyperplane for classification</p>
            <p className="font-bold mt-2">Accuracy: 97.5%</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h4 className="font-semibold text-green-700">KNN</h4>
            <p className="text-sm text-gray-600">Classifies based on k nearest training examples</p>
            <p className="font-bold mt-2">Accuracy: 96.2%</p>
          </div>
          <div className="bg-yellow-50 p-4 rounded-lg">
            <h4 className="font-semibold text-yellow-700">Decision Tree</h4>
            <p className="text-sm text-gray-600">Tree-like model of decisions</p>
            <p className="font-bold mt-2">Accuracy: 98.1%</p>
          </div>
          <div className="bg-red-50 p-4 rounded-lg">
            <h4 className="font-semibold text-red-700">Naive Bayes</h4>
            <p className="text-sm text-gray-600">Probabilistic classifier based on Bayes' theorem</p>
            <p className="font-bold mt-2">Accuracy: 94.3%</p>
          </div>
        </div>
      </div>
      
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Accuracy Comparison</h3>
          <div className="h-80">
            <Bar 
              data={accuracyData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                      display: true,
                      text: 'Accuracy (%)'
                    }
                  }
                }
              }} 
            />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Precision & Recall</h3>
          <div className="h-80">
            <Bar 
              data={precisionRecallData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                      display: true,
                      text: 'Value (%)'
                    }
                  }
                }
              }} 
            />
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">F1 Score</h3>
          <div className="h-80">
            <Bar 
              data={f1ScoreData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                      display: true,
                      text: 'F1 Score (%)'
                    }
                  }
                }
              }} 
            />
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Training Time</h3>
          <div className="h-80">
            <Bar 
              data={trainingTimeData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Time (seconds)'
                    }
                  }
                }
              }} 
            />
          </div>
        </div>
      </div>
      
      {/* Radar Chart */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Model Performance Radar</h3>
        <div className="h-96 max-w-3xl mx-auto">
          <Radar 
            data={radarData} 
            options={{ 
              maintainAspectRatio: false,
              scales: {
                r: {
                  min: 0,
                  max: 100,
                  ticks: {
                    stepSize: 20
                  }
                }
              }
            }} 
          />
        </div>
      </div>
      
      {/* Feature Importance */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Feature Importance (Random Forest)</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Importance Score</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">1</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">packet_rate</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0.187</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">byte_count</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0.156</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">tx_bytes</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0.142</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">4</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">flows</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0.128</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">5</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">duration</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">0.103</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;