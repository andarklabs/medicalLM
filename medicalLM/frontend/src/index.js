import React from "react";
import { render } from 'react-dom';
import { ChakraProvider } from "@chakra-ui/react";

import Header from "./components/Header";
import Todos from "./components/Todos";  // new

function App() {
  return (
    <ChakraProvider>
      <Header />
      <Todos />  {/* new */}
    </ChakraProvider>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)