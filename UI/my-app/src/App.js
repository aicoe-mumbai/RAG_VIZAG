// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login           from './components/Login';
import Dashboard       from './components/Dashboards';
import AdminDashboard  from './components/Admin';
import Landing         from './components/Landing';
import NotFound        from './components/NotFound';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() =>
    Boolean(sessionStorage.getItem('authToken'))
  );
  const [username, setUsername] = useState(() =>
    sessionStorage.getItem('userName') || ""
  );
  const [userUsecase, setUserUsecase] = useState(() =>
    sessionStorage.getItem('userUsecase') || ""
  );
  const [userRole, setUserRole] = useState(() =>
    sessionStorage.getItem('userRole') || ""
  );

  useEffect(() => {
    // Sync state with sessionStorage
    setIsAuthenticated(Boolean(sessionStorage.getItem('authToken')));
    setUsername(sessionStorage.getItem('userName') || "");
    setUserUsecase(sessionStorage.getItem('userUsecase') || "");
    setUserRole(sessionStorage.getItem('userRole') || "");
  }, []);
// console.log("hello",userRole)
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            isAuthenticated
              ? <Navigate to="/landing" replace />
              : <Navigate to="/login"   replace />
          }
        />

        <Route
          path="/login"
          element={
            isAuthenticated
              ? <Navigate to="/landing" replace />
              : (
                <Login
                  onLogin={() => {
                    setIsAuthenticated(true);
                    setUserUsecase(sessionStorage.getItem('userUsecase') || "");
                    setUserRole   (sessionStorage.getItem('userRole')    || "");
                  }}
                  setUsername={setUsername}
                />
              )
          }
        />

        <Route
          path="/landing"
          element={
            isAuthenticated
              ? <Landing username={username} />
              : <Navigate to="/login" replace />
          }
        />

        <Route
          path="/qa/dashboard"
          element={
            isAuthenticated && userUsecase === 'qa'
              ? <Dashboard
                  onLogout={() => setIsAuthenticated(false)}
                  username={username}
                  setUsername={setUsername}
                />
              : <Navigate to="/chat/dashboard" replace />
          }
        />
console.log(userRole)
        <Route
          path="/qa/admin"
          element={
            isAuthenticated && userRole == 'admin'
              ? <AdminDashboard />
              : <Navigate to="/login" replace />
          }
        />

        <Route
          path="/chat/dashboard"
          element={
            isAuthenticated
              ? <Dashboard
                  onLogout={() => setIsAuthenticated(false)}
                  username={username}
                  setUsername={setUsername}
                />
              : <Navigate to="/login" replace />
          }
        />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default App;
