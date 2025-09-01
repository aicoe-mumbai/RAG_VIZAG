import React, { useState, useEffect, useCallback } from "react";
import "./MilvusTable.css";

const MilvusTable = ({ collectionName, onClose }) => {
  const [data, setData] = useState([]);  
  const [filteredData, setFilteredData] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const apiUrl = process.env.REACT_APP_API_URL;

  const fetchData = useCallback(async (collectionName, page) => {
    setLoading(true);
    const token = sessionStorage.getItem("authToken");

    if (!token) {
      alert("No authentication token found");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(
        `${apiUrl}/api/milvus-data/${collectionName}/?page=${page}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }

      const result = await response.json();

      if (result.data.length > 0) {
        setData(result.data); 
        setFilteredData(result.data);
      } else {
        alert("No more data available");
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("Failed to fetch data.");
    }

    setLoading(false);
  }, [apiUrl]);

  useEffect(() => {
    if (collectionName) {
      fetchData(collectionName, page);
    }
  }, [collectionName, page, fetchData]);

  const handleSearch = (e) => {
    const query = e.target.value.toLowerCase();
    setSearchQuery(query);

    const filtered = data.filter((row) =>
      row.source.toLowerCase().includes(query)
    );
    setFilteredData(filtered);
  };

  const handleNextPage = () => {
    setPage((prevPage) => prevPage + 1);
  };

  const handlePreviousPage = () => {
    if (page > 1) {
      setPage((prevPage) => prevPage - 1);
    }
  };

  return (
    <div className="milvus-table">
      <div className="milvus-table-header">
        <button className="close-button1" onClick={onClose}>
          ✖
        </button>
      </div>
      <input
        type="text"
        placeholder="Search by source..."
        value={searchQuery}
        onChange={handleSearch}
        className="search-bar"
      />
      {loading ? (
        <p className="loading">Loading...</p>
      ) : (
        <div className="scrollable-table">
          <table className="milvus-table-table">
            <thead>
              <tr>
                <th>Serial No</th>
                <th>Source</th>
                <th>Page</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((row, index) => (
                <tr key={index}>
                  <td>{index + 1 + (page - 1) * 150}</td> {/* Correct serial number */}
                  <td>{row.source}</td>
                  <td>{row.page}</td>
                  <td>{row.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="pagination-container">
        <button
          onClick={handlePreviousPage}
          disabled={page === 1}
          className="pagination-button"
        >
          Previous
        </button>
        <span className="page-number">Page: {page}</span>
        <button onClick={handleNextPage} className="pagination-button">
          Next
        </button>
      </div>
    </div>
  );
};

export default MilvusTable;
