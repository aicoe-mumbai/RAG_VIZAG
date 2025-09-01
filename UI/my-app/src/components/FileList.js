import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./FileList.css";

const FileList = ({ collectionName, onClose, editMode, setProgress }) => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false); 
  const [searchQuery, setSearchQuery] = useState(""); 
  const [filteredFiles, setFilteredFiles] = useState([]); 
  const apiUrl = process.env.REACT_APP_API_URL; 
    const [source, setSource] = useState("");
    const navigate = useNavigate();
  
  useEffect(() => {
    const fetchFiles = async () => {
      setLoading(true);

      const token = sessionStorage.getItem("authToken");
      if (!token) {
        alert("No authentication token found");
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(
          `${apiUrl}/api/collections/${collectionName}/files/`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to fetch files");
        }

        const data = await response.json();
        setFiles(data.results); 
        setFilteredFiles(data.results);
      } catch (error) {
        console.error("Error fetching files:", error);
        alert("Some error occurred. Please login again.");
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('refreshToken');
        sessionStorage.clear();
        navigate("/login");
      } finally {
        setLoading(false);
      }
    };

    if (collectionName) {
      fetchFiles();
    } 
  }, [collectionName, apiUrl, navigate]);

  const fetchFiles = async () => {
    setLoading(true);

    const token = sessionStorage.getItem("authToken");
    if (!token) {
      alert("No authentication token found");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(
        `${apiUrl}/api/collections/${collectionName}/files/`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch files");
      }

      const data = await response.json();
      setFiles(data.results); 
      setFilteredFiles(data.results); 
    } catch (error) {
      console.error("Error fetching files:", error);
      alert("Some error occurred. Please login again.");
      sessionStorage.removeItem('authToken');
      sessionStorage.removeItem('refreshToken');
      sessionStorage.clear();
      navigate("/login");
    } finally {
      setLoading(false);
    }
  };
  
  // Handle search query
  const handleSearch = (e) => {
    const query = e.target.value.toLowerCase();
    setSearchQuery(query);

    // Filter files
    const filtered = files.filter((file) =>
      file.toLowerCase().includes(query)
    );
    setFilteredFiles(filtered);
  }; 

  // Handle file upload
  const handleFileUpload = async () => {
  
    if (!collectionName) {
      alert("Collection name is required.");
      return;
    }

    const token = sessionStorage.getItem("authToken");
    if (!token) {
      alert("No authentication token found");
      return;
    }

    const formData = new FormData();
    formData.append("name", collectionName); 
    formData.append("source", source);

    try {
      const progressInterval = setInterval(async () => {
        const progressResponse = await fetch(
          `${apiUrl}/api/collections/progress/`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (progressResponse.ok) {
          const progressData = await progressResponse.json();
          setProgress(progressData.message);

          if (progressData.message.includes("Upload completed.")) {
            clearInterval(progressInterval); 
            alert("Collection processing is complete.");
            setProgress("");
          }
        } else {
          clearInterval(progressInterval); 
          alert("Error fetching progress updates.");
        }
      }, 5000);

      alert("Collection is being processed. Check progress.");

      const response = await fetch(`${apiUrl}/api/collections/create_collection/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        alert(`Error: ${errorData.error}`);
        return;
      }

      alert("Files uploaded successfully");
      fetchFiles();
    } catch (error) {
      console.error("Error uploading files:", error);
    }
  };

   // Handle file deletion
  const handleDeleteFile = async (fileName) => {
    if (window.confirm("Are you sure you want to delete this file?")) {
      const token = sessionStorage.getItem("authToken");
      if (!token) {
        alert("No authentication token found");
        return;
      }
      const encodedFileName = encodeURIComponent(fileName);

      try {
        const response = await fetch(
          `${apiUrl}/api/collections/file-delete/${encodedFileName}/${collectionName}/`,
          {
            method: "DELETE",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to delete the file");
        }

        alert("File deleted successfully");
        setFiles((prevFiles) => prevFiles.filter((file) => file !== fileName));
        setFilteredFiles((prevFiltered) =>
          prevFiltered.filter((file) => file !== fileName)
        );
      } catch (error) {
        console.error("Error deleting file:", error);
        alert("Failed to delete the file");
      }
    }
  };

  return (
    <div className="file-list">
      <div className="file-list-header">
        <h2>{collectionName}</h2>
        <button className="close-button" onClick={onClose}>
          ✖
        </button>
      </div>
      <input
        type="text"
        placeholder="Search files..."
        value={searchQuery}
        onChange={handleSearch}
        className="search-bar"
      />
      {loading ? (
        <p>Loading files...</p>
      ) : filteredFiles.length > 0 ? (
        <div className="scrollable-table">
          <table className="file-table">
            <thead>
              <tr>
                <th>Serial No</th>
                <th>File Name</th>
                {editMode && <th>Action</th>}
              </tr>
            </thead>
            <tbody>
              {filteredFiles.map((file, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{file}</td>
                  {editMode && (
                    <td>
                      <button className="delete-button"  onClick={() => handleDeleteFile(file)} >
                        
                        Delete
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No files found</p>
      )}

      {editMode && (
        <div className="upload-section">
          {/* File input for single file selection */}
          <div>
            <label>Select Source: </label>
            <input
                type="text"
                className="input-source"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                required
            />
          </div>

          {/* Upload button */}
          <button onClick={handleFileUpload} className="upload-button">
            Upload 
          </button>
        </div>
      )}
    </div>
  );
};

export default FileList;
