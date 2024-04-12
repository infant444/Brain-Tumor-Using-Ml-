import router from 'express';
import asynchandler from 'express-async-handler'
import axios from 'axios'
import auth from '../middleware/auth.mid';
import { ChatRoomModel, chatroom } from '../model/chat.model';
import { ObjectId } from 'mongodb';

const rout=router(); 
rout.use(auth);

rout.get("/getchat/:userid",asynchandler(
    async(req,res)=>{
        const Chatroom=await ChatRoomModel.find({userid:req.params.userid});
        res.send(Chatroom);
    }
))
rout.post("/postchat",asynchandler(
    async(req,res)=>{
        const {name,send,time,userid}=req.body;
       const data:any={
        "send":send
        }
        const response = await axios.post("http://127.0.0.1:8000/api/chat", data)
        const result = response['data']
        const charrom:chatroom={
            name,
            send,
            responce: result['respons'],
            time,
            userid
        }
        const chat=await ChatRoomModel.create(charrom);
        if(chat){
            const chats=await ChatRoomModel.find({userid:userid});
            res.send(chats)
        }
    }
))
export default rout;