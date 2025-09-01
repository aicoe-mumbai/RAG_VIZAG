// src/components/Landing.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Landing.css';

function Landing({ username }) {
  const navigate = useNavigate();
  const [userUsecase, setUserUsecase] = useState(null);
  const [userRole,    setUserRole]    = useState(null);

  useEffect(() => {
    setUserUsecase(
      sessionStorage.getItem('userUsecase')?.toLowerCase() || null
    );
    setUserRole(sessionStorage.getItem('userRole') || null);
  }, []);

  const handleButtonClick = (mode) => {
    sessionStorage.setItem('lastMode', mode);
    if (mode === 'qa') {
      navigate('/qa/dashboard');
    } else if (mode === 'chat')  {
      const authToken = sessionStorage.getItem('authToken');
      const userName = sessionStorage.getItem('userName');
      const url = `http://172.16.34.231:8443/?authToken=${encodeURIComponent(authToken)}&userName=${encodeURIComponent(userName)}`;
      window.location.href = url;
    }
  };

  return (
    <div className="landing-container">
      <div className="landing-content">
        <h1>Welcome, {username}!</h1>
        <p>Select an option to get started</p>
        <div className="button-container">
          {userUsecase === 'qa' && (
            <button
              className="landing-button"
              onClick={() => handleButtonClick('qa')}
            >
              QA
            </button>
          )}
          <button
            className="landing-button"
            onClick={() => handleButtonClick('chat')}
          >
            Chat
          </button>
          {userRole === 'admin' && (
            <button
              className="landing-button"
              onClick={() => navigate('/qa/admin')}
            >
              Admin
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default Landing;