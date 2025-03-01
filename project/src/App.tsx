import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, BarChart3, Network, Database, Cpu } from 'lucide-react';
import Dashboard from './components/Dashboard';
import ModelComparison from './components/ModelComparison';
import DatasetInfo from './components/DatasetInfo';
import ControllerStatus from './components/ControllerStatus';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [controllerStatus, setControllerStatus] = useState('offline');
  
  // Simulate controller status check
  useEffect(() => {
    const interval = setInterval(() => {
      // In a real implementation, this would check the actual controller status
      setControllerStatus(Math.random() > 0.2 ? 'online' : 'offline');
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-indigo-700 text-white shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Shield size={28} />
            <h1 className="text-2xl font-bold">DDoS Detection in SDN</h1>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`inline-block w-3 h-3 rounded-full ${controllerStatus === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
            <span>Controller: {controllerStatus}</span>
          </div>
        </div>
      </header>
      
      {/* Navigation */}
      <nav className="bg-white shadow-md">
        <div className="container mx-auto px-4">
          <ul className="flex space-x-1">
            <li>
              <button 
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-3 flex items-center space-x-1 ${activeTab === 'dashboard' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600 hover:text-indigo-600'}`}
              >
                <BarChart3 size={18} />
                <span>Dashboard</span>
              </button>
            </li>
            <li>
              <button 
                onClick={() => setActiveTab('models')}
                className={`px-4 py-3 flex items-center space-x-1 ${activeTab === 'models' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600 hover:text-indigo-600'}`}
              >
                <Cpu size={18} />
                <span>ML Models</span>
              </button>
            </li>
            <li>
              <button 
                onClick={() => setActiveTab('dataset')}
                className={`px-4 py-3 flex items-center space-x-1 ${activeTab === 'dataset' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600 hover:text-indigo-600'}`}
              >
                <Database size={18} />
                <span>Dataset</span>
              </button>
            </li>
            <li>
              <button 
                onClick={() => setActiveTab('controller')}
                className={`px-4 py-3 flex items-center space-x-1 ${activeTab === 'controller' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-600 hover:text-indigo-600'}`}
              >
                <Network size={18} />
                <span>POX Controller</span>
              </button>
            </li>
          </ul>
        </div>
      </nav>
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'models' && <ModelComparison />}
        {activeTab === 'dataset' && <DatasetInfo />}
        {activeTab === 'controller' && <ControllerStatus status={controllerStatus} />}
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 text-white py-4">
        <div className="container mx-auto px-4 text-center">
          <p>DDoS Attack Detection in SDN using ML Techniques &copy; 2025</p>
        </div>
      </footer>
    </div>
  );
}

export default App;