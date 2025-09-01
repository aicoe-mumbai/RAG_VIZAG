
import React, { useState, useEffect } from "react";
import "./ChooseCollection.css";
import { useNavigate } from "react-router-dom";


const ChooseCollection = () => {
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [currentCollection, setCurrentCollection] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const token = sessionStorage.getItem("authToken");

  useEffect(() => {
    const fetchCollections = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/api/collections/`, {
          headers: {
            "Authorization": `Bearer ${token}`, 
          },
        });
        if (!response.ok) {
          throw new Error("Failed to fetch collections.");
        }
        const data = await response.json();
        setCollections(data.collections); 
      } catch (error) {
        alert("Some error occurred. Please login again.");
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('refreshToken');
        sessionStorage.clear();
        navigate("/login");
      }
    };

    const fetchCurrentCollection = async () => {
      try {
        console.log("hello harshit")
        const response = await fetch(`${process.env.REACT_APP_API_URL}/api/current-using-collection/`, {
          headers: {
            "Authorization": `Bearer ${token}`, 
          },
        });
        console.log("hello harshit", response)
        if (!response.ok) {
          throw new Error("Failed to fetch current collection.");
        }
        const data = await response.json();
        setCurrentCollection(data.current_using_collection || "No collection selected");
      } catch (error) {
        console.error("Error fetching current collection:", error);
        alert("Some error occurred. Please login again." );
        // sessionStorage.removeItem('authToken');
        // sessionStorage.removeItem('refreshToken');
        // sessionStorage.clear();
        // navigate("/login");
      }
    };

    fetchCollections();
    fetchCurrentCollection();
  }, [token, navigate]); 

  // Handle dropdown change
  const handleCollectionChange = (e) => {
    setSelectedCollection(e.target.value);
  };

  // Handle "Next" button click
  const handleChangeClick = async () => {
    if (selectedCollection === "") {
      alert("Please select a collection first!");
      return;
    }

    const userConfirmed = window.confirm(
      `Are you sure you want to change the current collection to "${selectedCollection}"?`
    );

    if (userConfirmed) {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/api/update-current-collection/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`, 
          },
          body: JSON.stringify({ current_using_collection: selectedCollection }),
        });

        if (response.ok) {
          setCurrentCollection(selectedCollection);
          alert("Successfully changed collection!");
        } else {
          alert("Failed to change collection. Please try again.");
        }
      } catch (error) {
        alert("An error occurred while updating the collection.");
      }
    }
  };
  const restartServer = async () => {
    try {
      const userConfirmed = window.confirm("Are you sure you want to restart the server?");
  
      if (!userConfirmed) {
        alert("Server restart canceled.");
        return; 
      }
      
      if (!token) {
        alert("No authentication token found. Please log in again.");
        return;
      }
      setLoading(true);
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/restart-server/`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
    
      const data = await response.json();
    
      if (response.ok) {
        setLoading(false);
        alert("Successfully restarted."); 
      } else {
        alert(`Failed to restart server: ${data.message}`);
      }
    } catch (error) {
      console.error('Error restarting the server:', error);
      setLoading(false);
      alert("Successfully restarted..");
    } 
  };
  
  
  return (
    <div>
     {loading && (
        <div className="loading-gif" >
          <img className="load-img" src="/images/loading_gif.gif" alt="Loading..." />
        </div>
      )}
<div className="collection-container">
  <div className="current-collection">
    <span>Current Selected Collection: </span>
    <span className="sp">{currentCollection}</span> {/* Display current collection */}
  </div>

  <div className="collection-selection">
    <label className="collection-label">Change Collection</label>
    <select 
      className="collection-dropdown" 
      value={selectedCollection} 
      onChange={handleCollectionChange}
    >
      <option value="">--Select a Collection--</option>
      {collections.map((collection, index) => (
        <option key={index} value={collection}>
          {collection}
        </option>
      ))}
    </select>
  </div>

  <div className="button-group">
    <button 
      className={`button ${!selectedCollection ? "button-disabled" : ""}`} 
      onClick={handleChangeClick} 
      disabled={!selectedCollection}
    >
      Change
    </button>
    <button className="button restart-button" onClick={restartServer}>
      Restart Server
    </button>

    <button className="dashboard-btn" onClick={() => navigate("/qa/dashboard")}>
          Go to Dashboard
        </button>
  </div>
</div> 


    </div>
  );
};

export default ChooseCollection;
