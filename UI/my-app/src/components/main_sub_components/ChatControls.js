// ChatControls.js
import React from 'react';
import './ChatControls.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons';
 
function ChatControls({ prompt, handleInputChange, handleKeyPress, handleSendPrompt, handleNewChat }) {
  return (
    <div className="chat-controls">
      <div className="message-input-container">
        <button className="new-chat-button" onClick={handleNewChat}>New Chat</button>
        <input 
          type="text" 
          placeholder="Send a message..." 
          className="message-input" 
          value={prompt}
          onChange={handleInputChange}  
          onKeyDown={handleKeyPress} 
          style={{ maxHeight: '200px', overflowY: 'auto', resize: 'none' }} 
          rows={1}
        />
        <button className="send-button" onClick={() => handleSendPrompt(prompt)}> <FontAwesomeIcon icon={faPaperPlane} /></button>
      </div>
    </div>
  );
}

export default ChatControls;
