import React from 'react';
import { Database, FileText, BarChart } from 'lucide-react';

const DatasetInfo = () => {
  // Dataset features from the image
  const features = [
    { id: 1, name: 'dt', description: 'Decision time' },
    { id: 2, name: 'switch', description: 'Switch identifier' },
    { id: 3, name: 'src', description: 'Source address' },
    { id: 4, name: 'dst', description: 'Destination address' },
    { id: 5, name: 'pkt_count', description: 'Packet count' },
    { id: 6, name: 'byte_count', description: 'Byte count' },
    { id: 7, name: 'duration', description: 'Duration' },
    { id: 8, name: 'duration_nsec', description: 'Duration in nanoseconds' },
    { id: 9, name: 'total_duration', description: 'Total duration' },
    { id: 10, name: 'flows', description: 'Number of flows' },
    { id: 11, name: 'pkt_rate', description: 'Packet rate' },
    { id: 12, name: 'pair_flow', description: 'Pair flow' },
    { id: 13, name: 'protocol', description: 'Protocol used' },
    { id: 14, name: 'port_no', description: 'Port number' },
    { id: 15, name: 'tx_bytes', description: 'Transmitted bytes' },
    { id: 16, name: 'rx_bytes', description: 'Received bytes' },
    { id: 17, name: 'tx_kbps', description: 'Transmitted kilobits per second' },
    { id: 18, name: 'rx_kbps', description: 'Received kilobits per second' },
    { id: 19, name: 'tot_kbps', description: 'Total kilobits per second' },
    { id: 20, name: 'label', description: 'Classification label (normal/attack)' },
    { id: 21, name: 'packetins', description: 'Packet ins' },
    { id: 22, name: 'byte_per_flow', description: 'Bytes per flow' },
    { id: 23, name: 'packet_per_flow', description: 'Packets per flow' }
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Dataset Information</h2>
      
      {/* Dataset Overview */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-start">
          <Database className="text-indigo-600 mr-4" size={24} />
          <div>
            <h3 className="text-lg font-semibold mb-2">Dataset Overview</h3>
            <p className="text-gray-600 mb-4">
              The dataset contains network traffic data collected from an SDN environment, with both normal traffic and DDoS attack traffic.
              It consists of 104,345 rows and 23 features that capture various aspects of network traffic.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Total Records</p>
                <p className="text-xl font-bold">104,345</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Features</p>
                <p className="text-xl font-bold">23</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-500">Class Distribution</p>
                <p className="text-xl font-bold">65% Normal, 35% Attack</p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Feature List */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-start mb-4">
          <FileText className="text-indigo-600 mr-4" size={24} />
          <div>
            <h3 className="text-lg font-semibold">Dataset Features</h3>
            <p className="text-gray-600">
              The dataset includes 23 features extracted from network traffic, capturing various aspects of traffic behavior.
            </p>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature Name</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {features.map((feature) => (
                <tr key={feature.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{feature.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{feature.name}</td>
                  <td className="px-6 py-4 text-sm text-gray-500">{feature.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Data Preprocessing */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-start">
          <BarChart className="text-indigo-600 mr-4" size={24} />
          <div>
            <h3 className="text-lg font-semibold mb-2">Data Preprocessing</h3>
            <p className="text-gray-600 mb-4">
              Before training the machine learning models, the dataset underwent several preprocessing steps to ensure optimal performance.
            </p>
            
            <h4 className="font-medium text-gray-800 mt-4 mb-2">Preprocessing Steps:</h4>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Missing value handling: Any missing values were imputed using appropriate methods</li>
              <li>Feature scaling: All features were normalized to have a mean of 0 and standard deviation of 1</li>
              <li>Feature selection: Correlation analysis was performed to identify and remove redundant features</li>
              <li>Outlier detection: Statistical methods were used to identify and handle outliers</li>
              <li>Class balancing: Techniques like SMOTE were applied to handle class imbalance</li>
              <li>Train-test split: Data was split into 70% training, 15% validation, and 15% test sets</li>
            </ul>
            
            <h4 className="font-medium text-gray-800 mt-4 mb-2">Feature Engineering:</h4>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Derived features: Additional features were created from existing ones to capture complex patterns</li>
              <li>Temporal features: Time-based features were extracted to capture traffic patterns over time</li>
              <li>Statistical features: Statistical measures were computed over sliding windows</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DatasetInfo;