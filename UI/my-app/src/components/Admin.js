import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import FileList from "./FileList";
import "./Admin.css";
import MilvusTable from "./MilvusTable";
import ChooseCollection from "./ChooseCollection";

const AdminDashboard = () => {
  const [collections, setCollections] = useState([]);
  const [selectedCollection, setSelectedCollection] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [isCreatingCollection, setIsCreatingCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState("");
  const [isEditMode, setIsEditMode] = useState(false);
  const [progress, setProgress] = useState(""); // For real-time progress display
  const [isViewingDetails, setIsViewingDetails] = useState(false);
  const apiUrl = process.env.REACT_APP_API_URL;
  const [source, setSource] = useState("");
  const navigate = useNavigate();


  useEffect(() => {
    const fetchCollections = async () => {
      setLoading(true);
      const token = sessionStorage.getItem("authToken");
      if (!token) {
        alert("No authentication token found");
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`${apiUrl}/api/collections/`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error("Failed to fetch collections");
        }

        const data = await response.json();
        setCollections(data.collections);
        console.log(data.username)

        // if(data.username != "1234"){
        //   navigate('/login');
        //   sessionStorage.clear();
        // }
      
      } catch (error) {
        console.log("Token", token)
        console.log(sessionStorage)
        console.error("Error fetching collections:", error);
        alert("Some error occurred. Please login again.", error);
        navigate("/login");
      } finally {
        setLoading(false);
      }
    };

    fetchCollections();
  }, [apiUrl, navigate]);

  const fetchCollections = async () => {
    setLoading(true);
    const token = sessionStorage.getItem("authToken");
    if (!token) {
      alert("No authentication token found");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/api/collections/`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch collections");
      }

      const data = await response.json();
      setCollections(data.collections);
    } catch (error) {
      console.error("Error fetching collections:", error);
      alert("Some error occurred. Please login again.");
      navigate("/login");
    } finally {
      setLoading(false);
    }
  };

  const handleCollectionClick = (collectionName) => {
    setSelectedCollection(collectionName);
    setIsEditMode(false);
    setIsCreatingCollection(false);
    setShowModal(true);
    setIsViewingDetails(false);


  };
  const handleDetails = (collection) => {
    setSelectedCollection(collection);
    setIsViewingDetails(true);
    setIsEditMode(false);
    setShowModal(true);
    setIsCreatingCollection(false);

  };

  const handleEdit = (collectionName) => {
    setSelectedCollection(collectionName);
    setIsEditMode(true);
    setIsCreatingCollection(false);
    setShowModal(true);
  };

  const handleDelete = async (collectionName) => {
    if (window.confirm(`Are you sure you want to delete ${collectionName}?`)) {
      const token = sessionStorage.getItem("authToken");
      if (!token) {
        alert("No authentication token found");
        return;
      }

      try {
        const response = await fetch(
          `${apiUrl}/api/collections/${collectionName}/delete/`,
          {
            method: "DELETE",
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to delete the collection");
        }

        setCollections((prevCollections) =>
          prevCollections.filter((collection) => collection !== collectionName)
        );
        alert(`${collectionName} deleted successfully`);
      } catch (error) {
        console.error("Error deleting collection:", error);
        alert("Failed to delete the collection");
      }
    }
  };

  const handleCreateCollection = async (e) => {
    e.preventDefault();
    setShowModal(false);

    const token = sessionStorage.getItem("authToken");
    if (!token) {
      alert("No authentication token found");
      return;
    }

    if (!newCollectionName) {
      alert("Collection name is required");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("name", newCollectionName);
      formData.append("source", source);

      // Poll for progress updates
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
            setNewCollectionName("");
            await fetchCollections();
          }
        } else {
          clearInterval(progressInterval);
          alert("Error fetching progress updates.");
        }
      }, 5000);

      alert("Collection is being processed. Check progress.");

      // Upload the files
      const uploadResponse = await fetch(
        `${apiUrl}/api/collections/create_collection/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      );

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        alert(`Error: ${errorData.error}`);
        return;
      }

    } catch (error) {
      console.error("Error creating collection:", error);
    }
  };

  return (
    <div className="admin-dashboard">
      <div className="header">
        <h1>Collections</h1>
        <button
          className="create-btn"
          onClick={() => {
            setShowModal(true);
            setIsCreatingCollection(true);
            setSelectedCollection(null);
          }}
        >
          Create New Collection
        </button>
      </div>
      <div><ChooseCollection /></div>

      {loading ? (
        <p>Loading collections...</p>
      ) : collections.length > 0 ? (
        <div className="card-container-admin">
          {collections.map((collection, index) => (
            <div key={index} className="card">
              <h3 className="collection-name-click" onClick={() => handleCollectionClick(collection)}>
                {collection}
              </h3>
              <div className="card-actions">
                <button
                  onClick={() => handleEdit(collection)}
                  className="edit-button"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDetails(collection)}
                  className="details-button"
                >
                  Details
                </button>
                <button
                  onClick={() => handleDelete(collection)}
                  className="delete-button"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p>No collections available</p>
      )}


      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            {isCreatingCollection ? (
              <form onSubmit={handleCreateCollection}>
                <label>
                  Collection Name:
                  <input
                    type="text"
                    className="collection-name-input"
                    value={newCollectionName}
                    onChange={(e) => setNewCollectionName(e.target.value)}
                    required
                  />
                </label>
                <br />
                <label>
                  Source:
                  <input
                    type="text"
                    className="create-source-input"
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    required

                  />
                </label>

                <br />

                <div className="modal-actions">
                  <button type="button" className="cancel-btn" onClick={() => setShowModal(false)}>
                    Cancel
                  </button>
                  <button type="submit" className="submit-bttn">Submit</button>
                </div>
              </form>
            ) : (
              <div>

                {isEditMode ? (
                  <FileList
                    collectionName={selectedCollection}
                    onClose={() => setShowModal(false)}
                    editMode={true}
                    setProgress={setProgress}
                  />
                ) : isViewingDetails ? (
                  <div>
                    <h2>Details for {selectedCollection}</h2>
                    <MilvusTable
                      collectionName={selectedCollection}
                      onClose={() => setShowModal(false)}
                    />
                  </div>
                ) : (
                  <FileList
                    collectionName={selectedCollection}
                    onClose={() => setShowModal(false)}
                    editMode={false}
                  />
                )}

              </div>
            )}
          </div>
        </div>
      )}

      {/* Single Progress Display */}
      {progress && (
        <div className="progress-container">
          <p>{progress}</p>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
