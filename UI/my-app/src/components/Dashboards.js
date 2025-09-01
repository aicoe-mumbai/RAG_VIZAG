// src/Dashboard.js
import React from 'react';
import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import './App.css';
import LeftSidebar from './LeftSidebar';
import MainContent from './MainContent.js';
import ThemeProvider from './main_sub_components/ThemeProvider'; 

function Dashboard({ onLogout, username, setUsername }) {
  const location = useLocation();
  const currentRouteType = location.pathname.split('/')[1]; 
  console.log("Current Route Type:", currentRouteType); // Debugging line
  const [history, setHistory] = useState({ today: [], yesterday: [], last_week: [], last_month: [] });
  const [selectedSessionId, setSelectedSessionId] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]); 
  const [userfile, ansetuserfile] = useState(null);

  // Callback function to handle history clicks
  const handleHistoryClick = (session_id) => {
    setSelectedSessionId(session_id); 
  }; 

  // Callback to handle file selection from LeftSidebar
  const handleFileSelection = (files) => {
    setSelectedFiles(files); 
  };

  return (
    <ThemeProvider>
      <div className="App">
        <div className="container">
          <LeftSidebar 
            history={history} 
            onLogout={onLogout} 
            userfile={userfile} 
            ansetuserfile={ansetuserfile}
            onHistoryClick={handleHistoryClick} 
            onFileSelect={handleFileSelection} 
            username={username} 
            setUsername={setUsername} 
            mode={currentRouteType} // Pass mode to LeftSidebar
          />
          <MainContent 
            setHistory={setHistory} 
            selectedSessionId={selectedSessionId}
            resetSelectedSessionId={() => setSelectedSessionId(null)}  
            selectedFiles={selectedFiles} 
            userfile={userfile}
            mode={currentRouteType} // Pass mode to MainContent
          />
        </div>
      </div>
    </ThemeProvider>
  );
}

export default Dashboard;