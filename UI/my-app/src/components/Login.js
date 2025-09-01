import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

function Login({ onLogin, setUsername }) {
  const [username, setUsernameLocal] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const apiUrl = process.env.REACT_APP_API_URL;

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${apiUrl}/api/login/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      // Store tokens & username
      sessionStorage.setItem('authToken',    data.access);
      sessionStorage.setItem('refreshToken', data.refresh);
      sessionStorage.setItem('userName',     data.username);
      sessionStorage.setItem('userUsecase',  data.usecase);
      sessionStorage.setItem('userRole',     data.role);

      // Update parent state
      onLogin();
      setUsername(data.username);

      // Redirect to landing
      navigate('/landing');
    } catch (error) {
      setError('Invalid credentials');
      console.error('Error during login:', error);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="login-page">
      <div className="login-container">
        <h2 className="login-title">Login</h2>
        {error && <p className="error-message">{error}</p>}
        <form className="login-form" onSubmit={handleLogin}>
          <input
            className="input-field"
            type="text"
            value={username}
            onChange={(e) => setUsernameLocal(e.target.value)}
            placeholder="PSNO@ltdic.com"
            required
          />
          <div className="password-field-container">
            <input
              className="input-field"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              required
            />
            <button
              type="button"
              className="password-toggle-btn"
              onClick={togglePasswordVisibility}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? '👁️' : '👁️‍🗨️'}
            </button>
          </div>
          <button className="submit-btn" type="submit">
            Login
          </button>
        </form>
      </div>
    </div>
  );
}

export default Login;
