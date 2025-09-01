import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './LeftSidebar.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHistory, faSignOutAlt, faCloudArrowUp, faTimes } from '@fortawesome/free-solid-svg-icons';
import Select from 'react-select';
import { Cascader } from "antd";

function LeftSidebar({ history, onHistoryClick, onFileSelect, onLogout, userfile, ansetuserfile, username, setUsername, mode }) {
  const navigate = useNavigate();

  // Helper function to match usecase properly:
  function usecaseMatchesMode(itemUsecase, mode) {
    if (mode === 'chat') {
      // Show entries with usecase 'chat' or null/undefined for chat mode
      return itemUsecase === 'chat' || itemUsecase === null || itemUsecase === undefined;
    } else {
      // For 'qa', show only exact 'qa'
      return itemUsecase === mode;
    }
  }

const filteredHistory = history;

  const [selectedOptions, setSelectedOptions] = useState([]);
  const [options, setOptions] = useState([]);
  const apiUrl = process.env.REACT_APP_API_URL;

  const userhandleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    ansetuserfile(file);
    onFileSelect([]);
    await userhandleupload(file);
  };

  const handleRemoveFile = () => {
    ansetuserfile(null);
  };

  const userhandleupload = async (fileParan) => {
    const token = sessionStorage.getItem("authToken");
    const fileToUpload = fileParan || userfile;
    const formData = new FormData();
    formData.append("file", fileToUpload);

    try {
      const response = await fetch(`${apiUrl}/api/upload_file_from_the_user/`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      alert("File uploaded successfully! OCR will not be supported only readable files are allowed currently (*pdf)");
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Failed to upload file. Please select a file and try again. Currently, only PDF files are supported. OCR files, XLSX, PPTX, TXT, DOCX, and CSV will be supported in the next phase");
      ansetuserfile(null);
    }
  };

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/documents/`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${sessionStorage.getItem('authToken')}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const data = await response.json();
          const formattedOptions = data['files'].map(doc => ({
            value: doc,
            label: doc.substring(doc.lastIndexOf('/') + 1),
          }));
          setOptions(formattedOptions);
          setUsername(data.username);
        } else {
          console.error('Failed to fetch options');
        }
      } catch (error) {
        console.error('Error fetching options:', error);
      }
    };

    fetchOptions();
  }, [apiUrl, navigate]);

  const handleChange = (selected) => {
    setSelectedOptions(selected);
    const selectedFiles = selected ? selected.map(option => option.value) : [];
    onFileSelect(selectedFiles);
    ansetuserfile(null);
  };

  const handleLogout = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/logout/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionStorage.getItem('authToken')}`,
        },
        body: JSON.stringify({
          refresh: sessionStorage.getItem('refreshToken'),
        }),
      });

      if (response.ok) {
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('refreshToken');
        sessionStorage.clear();

        onLogout();
        navigate('/login');
      }
    } catch (error) {
      console.error('Error during logout:', error);
    }
  };

  return (
    <div className="left-sidebar">
      <div className="new">
        <img src="/images/L&T PES - Linear Logo - Black.jpg" alt="A descriptive alt text" className='lnt-logo' />
      </div>
      {mode === 'qa' && (
        <div className="multi-select">
          {!userfile && (<Select
            isMulti
            name="options"
            options={options}
            value={selectedOptions}
            onChange={handleChange}
            className="multi-select"
          />)}
        </div>
      )}

      <label className="custom-file-input">
        <FontAwesomeIcon icon={faCloudArrowUp} className="upload" />
        Upload document
        <input type="file" accept='.pdf,.csv,.xlsx,.pptx' onChange={userhandleFileChange} />
      </label>

      {userfile && (
        <div style={{
          marginTop: "3px",
          padding: "0.5rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis"
        }}>

          <span style={{ flex: "1", minWidth: "0", overflow: "hidden", textOverflow: "ellipsis" }}>
            {userfile.name}</span>
          <button
            onClick={handleRemoveFile}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              color: "red",
              fontSize: "1rem"
            }}
            title="Deselect file"
          >
            <FontAwesomeIcon icon={faTimes} />
          </button>
        </div>
      )}
      <div className="history-section">
        <h4>Today</h4>
        <ul>
          {filteredHistory.today.map((item) => (
            <li key={item.id} onClick={() => onHistoryClick(item.session_id)} title={item.prompt}>
              <FontAwesomeIcon icon={faHistory} className="history-icon" />
              {item.prompt}
            </li>
          ))}
        </ul>

        <h4>Yesterday</h4>
        <ul>
          {filteredHistory.yesterday.map((item) => (
            <li key={item.id} onClick={() => onHistoryClick(item.session_id)} title={item.prompt}>
              <FontAwesomeIcon icon={faHistory} className="history-icon" />
              {item.prompt}
            </li>
          ))}
        </ul>

        <h4>Last Week</h4>
        <ul>
          {filteredHistory.last_week.map((item) => (
            <li key={item.id} onClick={() => onHistoryClick(item.session_id)} title={item.prompt}>
              <FontAwesomeIcon icon={faHistory} className="history-icon" />
              {item.prompt}
            </li>
          ))}
        </ul>

        <h4>Last Month</h4>
        <ul>
          {filteredHistory.last_month.map((item) => (
            <li key={item.id} onClick={() => onHistoryClick(item.session_id)} title={item.prompt}>
              <FontAwesomeIcon icon={faHistory} className="history-icon" />
              {item.prompt}
            </li>
          ))}
        </ul>
      </div>

      <div className="progress-bar"></div>

      <div className="user-card-container">
        <div className="user-details">
          <img
            src="/images/ai-technology.png"
            alt="User Avatar"
            className="user_avatar"
          />
          <span className="user-name">{username || 'Guest'}</span>
          <div className="user-actions">

            {/* Logout Icon with Tooltip */}
            <div className="icon-with-tooltip">
              <FontAwesomeIcon
                icon={faSignOutAlt}
                className="user-icon"
                onClick={handleLogout}
              />
              <span className="tooltip-text">Logout</span>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}

export default LeftSidebar;
