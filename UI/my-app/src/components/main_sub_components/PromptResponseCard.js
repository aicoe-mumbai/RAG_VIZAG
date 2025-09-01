import React, { useState } from 'react';
import './PromptResponseCard.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faThumbsUp, faThumbsDown, faCopy, faEdit, faArrowDown } from '@fortawesome/free-solid-svg-icons';
import {marked} from 'marked'

function renderMarkdown(text){
  return marked(text || '');
}
function PromptResponseCard({ item, index, editingIndex, setEditingIndex, handleSubmitEditedPrompt, onSendPrompt, mode }) {
  const [editedPrompt, setEditedPrompt] = useState(item.prompt);
  const [showCopyIcon, setShowCopyIcon] = useState(true);
  const [isThumbsUpActive, setIsThumbsUpActive] = useState(false);
  const [isThumbsDownActive, setIsThumbsDownActive] = useState(false);
  const [comment, setComment] = useState('');
  const [isCommentSubmitted, setIsCommentSubmitted] = useState(false);
  const [isThinkingVisible, setIsThinkingVisible] = useState(true); // Automatically show "Thinking"
  const apiUrl = process.env.REACT_APP_API_URL;

  const handleEditClick = () => {
    setEditingIndex(index);
    setEditedPrompt(item.prompt);
  };

  const handleCopyToClipboard = () => {
    if (item.response) {
      const formattedText = item.response.replace(/\s+/g, ' ').trim();
      navigator.clipboard.writeText(formattedText)
        .then(() => {
          setShowCopyIcon(false);
          setTimeout(() => setShowCopyIcon(true), 2000);
        })
        .catch((err) => console.error("Failed to copy text: ", err));
    }
  };

  const handleThumbsUpClick = async () => {
    setIsThumbsUpActive(true);
    if (isThumbsDownActive) {
      setIsThumbsDownActive(false);
      setIsCommentSubmitted(false);
    }
    const token = sessionStorage.getItem('authToken');

    try {
      const status = "satisfied"
      const response = await fetch(`${apiUrl}api/mark_satisfied_or_unsatisfied/${item.id}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ status })
      });

      if (response.ok) {
      } else {
        console.error('Failed to mark as satisfied:', response.statusText);
      }
    } catch (error) {
      console.error('Error marking as satisfied:', error);
    }
  };

  const handleThumbsDownClick = async () => {
    setIsThumbsDownActive(true);
    setIsThumbsUpActive(false);
    setIsCommentSubmitted(false);
    const token = sessionStorage.getItem('authToken');

    try {
      const status = "unsatisfied"
      const response = await fetch(`${apiUrl}/api/mark_satisfied_or_unsatisfied/${item.id}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ status })
      });

      if (response.ok) {
        const data = await response.json();
        console.log(data.message);
      } else {
        console.error('Failed to mark as unsatisfied:', response.statusText);
      }
    } catch (error) {
      console.error('Error marking as unsatisfied:', error);
    }
  };

  const handleCommentSubmit = async () => {
    const token = sessionStorage.getItem('authToken');

    try {
      const response = await fetch(`${apiUrl}/api/save-comment/${item.id}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ comments: comment })
      });

      if (response.ok) {
        setIsCommentSubmitted(true);
        setComment('');
      } else {
        console.error('Failed to submit comment:', response.statusText);
      }
    } catch (error) {
      console.error('Error submitting comment:', error);
    }
  };

  const handlePdfClick = async (fileName, pageNumber) => {
    const token = sessionStorage.getItem('authToken');

    const encodedFileName = encodeURIComponent(fileName);
    try {
      const response = await fetch(`${apiUrl}/api/serve-file/${encodedFileName}/${pageNumber}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
      });

      if (response.ok) {
        const blob = await response.blob();
        const fileUrl = URL.createObjectURL(blob);

        if (fileName.endsWith(".pdf") || fileName.endsWith(".docx")) {
          window.open(`${fileUrl}#page=${pageNumber}`, '_blank');
        } else {
          const link = document.createElement('a');
          link.href = fileUrl;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      } else {
        console.error("Error fetching PDF:", response.statusText);
      }
    } catch (error) {
      console.error("Error fetching PDF:", error);
    }
  };

  const processResponseLinks = (responseText) => {
    const mainContent = responseText
    return {  mainContent };

  };

  const { mainContent } = processResponseLinks(item.response || "");


  const renderSourceLinks = (responseText) => {
    const processBold = (text) => {
      return text.split(/(\*{1,2}.*?\*{1,2})/).map((part,i)  => {
        if (part.startsWith("*") && part.endsWith("*")) {
          const content = part.slice (1,-1);
          return <strong key = {i}> {content}</strong>;
        }
        return part;
      });
    };
    
    // const processBold = (text) => {
    //   const parts = text.split(/(\*\*.*?\*\*)/);
    //   return parts.map((part, i) => {
    //     if (part.startsWith("*") && part.endsWith("*")) {
    //       return <strong key={i}>{part.slice(2, -2)}</strong>;
    //     }
    //     return part;
    //   });
    // };
    
    return responseText.split('\n\n').map((part, index) => {
      const updatedPart = part.split(/(Source:\s.*?\.[a-zA-Z0-9]+ \| Page:\s\d+)/gi).map((chunk, idx) => {
        const sourceMatch = chunk.match(/Source:\s(.*?)\s?\| Page:\s(\d+)/i);

        if (sourceMatch) {
          const [, filePath, pageNumber] = sourceMatch;

          const fileName = filePath.split('/').pop();
          const displayText = `Source: ${fileName} - Page: ${pageNumber}`;
          const encodedFileName = encodeURIComponent(filePath);

          return (
            <div key={fileName + pageNumber}>
              <a
                href={`/api/serve-file/${encodedFileName}/${pageNumber}/`}
                onClick={(e) => {
                  e.preventDefault();
                  handlePdfClick(fileName, pageNumber);
                }}
              >
                {displayText}
              </a>
            </div>
          );
        }
        return processBold(chunk);
      });

      return (
        <span key={index}>
          {updatedPart}
          {index !== responseText.split('\n\n').length - 1 && <br />}
        </span>
      );
    });
  };

  return (
    <div className="card-container">
      <div className="prompt-card">
        {editingIndex === index ? (
          <div className="edit-prompt-card">
            <input
              type="text"
              className="edit-prompt-input"
              value={editedPrompt}
              onChange={(e) => setEditedPrompt(e.target.value)}
            />
            <button className="save-edit-button" onClick={() => handleSubmitEditedPrompt(editedPrompt, index)}>Send</button>
            <button className="cancel-edit-button" onClick={() => setEditingIndex(null)}>Cancel</button>
          </div>
        ) : (
          <div className="prompt-section">
            <p>{item.prompt}</p>
            <span className="edit-icon" onClick={handleEditClick}>
              <FontAwesomeIcon icon={faEdit} />
            </span>
          </div>
        )}
      </div>
      <img
        src="/images/hacker.png"
        alt="Response Avatar"
        className="avatar2"
      />

      <div className="response-card">
        <div > {mainContent? renderSourceLinks(mainContent) : "Loading..."} </div>
        {mode === 'qa' && (
        <div className="pdf-button-container">
          <div className="continue-container">
            <FontAwesomeIcon icon={faArrowDown} className="arrow-icon" />
            <button
              className="continue-link"
              onClick={() => onSendPrompt("Continue")}
            >
              Continue
            </button>
          </div>
        </div>
        )}

        <div className='thumps'>
          <div className='copy-icon' onClick={handleCopyToClipboard}>
            {showCopyIcon ? (
              <div className="tooltip">
                {/* <FontAwesomeIcon icon={faCopy} /> */}
              </div>
            ) : (
              <span className="copied-message">Copied!</span>
            )}
          </div>
          <div className='thumpsup' onClick={handleThumbsUpClick}>
            <FontAwesomeIcon icon={faThumbsUp} className={isThumbsUpActive ? 'active' : ''} />
          </div>
          <div className='thumpsdown' onClick={handleThumbsDownClick}>
            <FontAwesomeIcon icon={faThumbsDown} className={isThumbsDownActive ? 'active' : ''} />
          </div>
        </div>

        {isThumbsDownActive && !isCommentSubmitted && (
          <div className="comment-box">
            <textarea
              placeholder="Type your comment here..."
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              className="comment-input"
            />
            <button onClick={handleCommentSubmit} className="comment-submit-button">Submit</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default PromptResponseCard;
