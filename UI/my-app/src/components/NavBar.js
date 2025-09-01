import './NavBar.css';

function NavBar() {   
  return (
    <header className="navbar">
      <h1 className="header-title">
        <span className='sub-head'><b>LUMINAR</b> </span>
        <span className="header-subtext"> developed by </span> <span className="header-sub">PES AICoE</span>
      </h1>
      
     <div className=''>
     <img src="/images/AICoE_logo.png" alt="A descriptive alt text" className='aicoe-logo' />
     </div>
    </header>
  );
}
 
export default NavBar;
