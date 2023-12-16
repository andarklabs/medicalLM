import React, { useEffect, useState } from "react";
import {
    Box,
    Button,
    Flex,
    Input,
    InputGroup,
    Modal,
    ModalBody,
    ModalCloseButton,
    ModalContent,
    ModalFooter,
    ModalHeader,
    ModalOverlay,
    Stack,
    Text,
    useDisclosure
} from "@chakra-ui/react";

const TodosContext = React.createContext({
    todos: [], fetchTodos: () => { }
})

export default function Todos() {
    const [todos, setTodos] = useState([])
    const fetchTodos = async () => {
        const response = await fetch("http://localhost:8000/todo")
        const todos = await response.json()
        setTodos(todos.data)
    }
    useEffect(() => {
        fetchTodos()
    }, [])
    return (
        <TodosContext.Provider value={{ todos, fetchTodos }}>
            <Stack spacing={5}>
                {todos.map((todo) => (
                    <b>{todo.item}</b>
                ))}
            </Stack>
        </TodosContext.Provider>
    )
}