import React from 'react';
import { Network, Server, Shield, AlertTriangle, CheckCircle } from 'lucide-react';

interface ControllerStatusProps {
  status: string;
}

const ControllerStatus: React.FC<ControllerStatusProps> = ({ status }) => {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">POX Controller Integration</h2>
      
      {/* Controller Status */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center mb-4">
          <Server className={status === 'online' ? 'text-green-500' : 'text-red-500'} size={24} />
          <h3 className="text-lg font-semibold ml-2">Controller Status</h3>
        </div>
        
        <div className="flex items-center p-4 bg-gray-50 rounded-lg">
          <div className={`w-4 h-4 rounded-full ${status === 'online' ? 'bg-green-500' : 'bg-red-500'} mr-2`}></div>
          <span className="font-medium">{status === 'online' ? 'Online' : 'Offline'}</span>
          <span className="ml-2 text-gray-500">- {status === 'online' ? 'Controller is running and processing traffic' : 'Controller is not running'}</span>
        </div>
        
        {status === 'online' && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 p-4 rounded-lg flex items-center">
              <CheckCircle className="text-green-500 mr-2" size={20} />
              <div>
                <p className="text-sm text-gray-500">Connected Switches</p>
                <p className="text-xl font-bold">5</p>
              </div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg flex items-center">
              <Network className="text-blue-500 mr-2" size={20} />
              <div>
                <p className="text-sm text-gray-500">Active Flows</p>
                <p className="text-xl font-bold">1,245</p>
              </div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg flex items-center">
              <AlertTriangle className="text-red-500 mr-2" size={20} />
              <div>
                <p className="text-sm text-gray-500">Blocked Attacks</p>
                <p className="text-xl font-bold">37</p>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Controller Architecture */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Controller Architecture</h3>
        <div className="flex flex-col md:flex-row">
          <div className="md:w-1/2 p-4">
            <p className="text-gray-600 mb-4">
              The POX controller is integrated with our machine learning models to detect and mitigate DDoS attacks in real-time.
              The controller monitors network traffic, extracts features, and uses the trained models to classify traffic as normal or attack.
            </p>
            <h4 className="font-medium text-gray-800 mt-4 mb-2">Key Components:</h4>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li><span className="font-medium">Traffic Monitor:</span> Collects traffic statistics from OpenFlow switches</li>
              <li><span className="font-medium">Feature Extractor:</span> Processes raw traffic data into feature vectors</li>
              <li><span className="font-medium">ML Classifier:</span> Uses the Random Forest model to classify traffic</li>
              <li><span className="font-medium">Flow Manager:</span> Installs flow rules to block malicious traffic</li>
              <li><span className="font-medium">API Server:</span> Provides REST API for monitoring and configuration</li>
            </ul>
          </div>
          <div className="md:w-1/2 p-4 flex justify-center items-center">
            <div className="bg-gray-50 p-6 rounded-lg w-full">
              <div className="text-center mb-4">
                <Shield className="inline-block text-indigo-600" size={40} />
                <h4 className="font-semibold mt-2">ML-Enhanced SDN Controller</h4>
              </div>
              <div className="space-y-2">
                <div className="bg-blue-100 p-2 rounded text-center">Traffic Monitoring</div>
                <div className="text-center">↓</div>
                <div className="bg-green-100 p-2 rounded text-center">Feature Extraction</div>
                <div className="text-center">↓</div>
                <div className="bg-purple-100 p-2 rounded text-center">Random Forest Classification</div>
                <div className="text-center">↓</div>
                <div className="bg-red-100 p-2 rounded text-center">Attack Mitigation</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Controller Commands */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Controller Commands</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Command</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono bg-gray-50">python src/pox/pox_controller.py</td>
                <td className="px-6 py-4 text-sm text-gray-500">Start the POX controller with ML-based DDoS detection</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono bg-gray-50">python src/ml/train_models.py</td>
                <td className="px-6 py-4 text-sm text-gray-500">Train all ML models on the dataset</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono bg-gray-50">python src/ml/evaluate_models.py</td>
                <td className="px-6 py-4 text-sm text-gray-500">Evaluate and compare all ML models</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-mono bg-gray-50">python src/tools/generate_traffic.py</td>
                <td className="px-6 py-4 text-sm text-gray-500">Generate test traffic (normal and attack) for testing</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ControllerStatus;