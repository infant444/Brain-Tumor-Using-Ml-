
import{connect, ConnectOptions} from 'mongoose';


export const dbconnect=()=>{
    connect(process.env.MONGODB_URL!,{
        
    }as ConnectOptions).then(
        ()=>console.log("connect successfully"),
        (error)=>console.log(error)
    )
}